"""FeatureCollection class for bookkeeping and calculation of time-series features.

Methods, next to `FeatureCollection.calculate()`, worth looking at: \n
* `FeatureCollection.serialize()` - serialize the FeatureCollection to a file
* `FeatureCollection.reduce()` - reduce the number of features after feature selection

"""

from __future__ import annotations

import warnings

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

import os
import traceback
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import dill
import numpy as np
import pandas as pd
from multiprocess import Pool
from tqdm.auto import tqdm

from ..features.function_wrapper import FuncWrapper
from ..utils.attribute_parsing import AttributeParser
from ..utils.data import flatten, to_list, to_series_list
from ..utils.logging import add_logging_handler, delete_logging_handlers
from ..utils.time import parse_time_arg, timedelta_to_str
from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .logger import logger
from .segmenter import StridedRolling, StridedRollingFactory
from .utils import _check_start_end_array, _determine_bounds


class FeatureCollection:
    """Create a FeatureCollection.

    Parameters
    ----------
    feature_descriptors : Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection, List[Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection]]], optional
        Initial (list of) feature(s) to add to collection, by default None

    Notes
    -----
    * The `series_name` property of the `FeatureDescriptor`s should **not withhold a "|"
      character**, since "|" is used to join the series names of features which use
      multiple series as input).<br>
      e.g.<br>
        * `ACC|x` is **not** allowed as series name, as this is ambiguous and could
          represent that this feature is constructed with a combination of the `ACC`
          and `x` signal.<br>
          Note that `max|feat` is allowed as feature output name.
    * Both the `series_name` and `output_name` property of the `FeatureDescriptor`s
      **should not withhold "__"** in its string representations. This constraint is
      mainly made for readability purposes.

    The two statements above will be asserted

    """

    def __init__(
        self,
        feature_descriptors: Optional[
            Union[
                FeatureDescriptor,
                MultipleFeatureDescriptors,
                FeatureCollection,
                List[
                    Union[
                        FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection
                    ]
                ],
            ]
        ] = None,
    ):
        # The feature collection is a dict with keys of type:
        #   tuple(tuple(str), float OR pd.timedelta)
        # The outer tuple's values correspond to (series_key(s), window)
        self._feature_desc_dict: Dict[
            Tuple[Tuple[str, ...], Union[float, pd.Timedelta]], List[FeatureDescriptor]
        ] = {}

        if feature_descriptors:
            self.add(feature_descriptors)

    def get_required_series(self) -> List[str]:
        """Return all required series names for this feature collection.

        Return the list of series names that are required in order to calculate all the
        features (defined by the `FeatureDescriptor` objects) of this feature
        collection.

        Returns
        -------
        List[str]
            List of all the required series names.

        """
        return list(
            set(flatten([fr_key[0] for fr_key in self._feature_desc_dict.keys()]))
        )

    def get_nb_output_features(self) -> int:
        """Return the number of output features in this feature collection.

        Returns
        -------
        int
            The number of output features in this feature collection.

        """
        fd_list: Iterable[FeatureDescriptor] = flatten(self._feature_desc_dict.values())
        return sum(fd.get_nb_output_features() for fd in fd_list)

    def _get_nb_output_features_without_window(self) -> int:
        """Return the number of output features in this feature collection, without
        using the window as a unique identifier.

        This is relevant for when the window value(s) are overridden by passing
        `segment_start_idxs` and `segment_end_idxs` to the `calculate` method.

        Returns
        -------
        int:
            The number of output features in this feature collection without using the
            window as a unique identifier.

        """
        return len(
            set(
                (series, o)
                for (series, _), fd_list in self._feature_desc_dict.items()
                for fd in fd_list
                for o in fd.function.output_names
            )
        )

    @staticmethod
    def _get_collection_key(
        feature: FeatureDescriptor,
    ) -> Tuple[Tuple[str, ...], Union[pd.Timedelta, float, None]]:
        # Note: `window` property can be either a pd.Timedelta or a float or None
        # assert feature.window is not None
        return feature.series_name, feature.window

    def _check_feature_descriptors(
        self,
        skip_none: bool,
        calc_stride: Optional[Union[float, pd.Timedelta, None]] = None,
    ):
        """Verify whether all added FeatureDescriptors imply the same-input data type.

        If this condition is not met, a warning will be raised.

        Parameters
        ----------
        skip_none: bool
            Whether to include None stride values in the checks.
        calc_stride: Union[float, pd.Timedelta, None], optional
            The `FeatureCollection.calculate` its stride argument, by default None.
            This stride takes precedence over a `FeatureDescriptor` its stride when
            it is not None.

        """
        dtype_set = set()
        for series_names, win in self._feature_desc_dict.keys():
            for fd in self._feature_desc_dict[(series_names, win)]:
                stride = calc_stride if calc_stride is not None else fd.stride
                if skip_none and stride is None:
                    dtype_set.add(AttributeParser.determine_type(win))
                else:
                    dtype_set.add(
                        AttributeParser.determine_type([win] + to_list(stride))
                    )

        if len(dtype_set) > 1:
            warnings.warn(
                "There are multiple FeatureDescriptor window-stride "
                + f"datatypes present in this FeatureCollection, i.e.: {dtype_set}",
                category=RuntimeWarning,
            )

    def _add_feature(self, feature: FeatureDescriptor):
        """Add a `FeatureDescriptor` instance to the collection.

        Parameters
        ----------
        feature : FeatureDescriptor
            The feature that will be added to this feature collection.

        """
        # Check whether the `|` is not present in the series
        assert not any("|" in s_name for s_name in feature.get_required_series())
        # Check whether the '__" is not present in the series and function output names
        assert not any(
            "__" in output_name for output_name in feature.function.output_names
        )
        assert not any("__" in s_name for s_name in feature.get_required_series())

        series_win_stride_key = self._get_collection_key(feature)
        if series_win_stride_key in self._feature_desc_dict.keys():
            added_output_names = flatten(
                f.function.output_names
                for f in self._feature_desc_dict[series_win_stride_key]
            )
            # Check that not a feature with the same output_name(s) is already added
            # for the series_win_stride_key
            assert not any(
                output_name in added_output_names
                for output_name in feature.function.output_names
            )
            self._feature_desc_dict[series_win_stride_key].append(feature)
        else:
            self._feature_desc_dict[series_win_stride_key] = [feature]

    def add(
        self,
        features: Union[
            FeatureDescriptor,
            MultipleFeatureDescriptors,
            FeatureCollection,
            List[
                Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection]
            ],
        ],
    ):
        """Add feature(s) to the FeatureCollection.

        Parameters
        ----------
        features : Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection, List[Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection]]]
            Feature(s) (containers) whose contained features will be added.

        Raises
        ------
        TypeError
            Raised when an item within `features` is not an instance of
            [`MultipleFeatureDescriptors`, `FeatureDescriptors`, `FeatureCollection`].

        """
        # Convert to list if necessary
        features = to_list(features)

        for feature in features:
            if isinstance(feature, MultipleFeatureDescriptors):
                self.add(feature.feature_descriptions)
            elif isinstance(feature, FeatureDescriptor):
                self._add_feature(feature)
            elif isinstance(feature, FeatureCollection):
                # List needs to be flattened
                self.add(list(flatten(feature._feature_desc_dict.values())))
            else:
                raise TypeError(f"type: {type(feature)} is not supported - {feature}")

        # After adding the features, check whether the descriptors are compatible
        self._check_feature_descriptors(skip_none=True)

    @staticmethod
    def _executor(idx: int):
        # global get_stroll_func
        stroll, function = get_stroll_func(idx)
        return stroll.apply_func(function)

    # def _get_stroll(self, kwargs):
    #     return StridedRollingFactory.get_segmenter(**kwargs)

    def _stroll_feat_generator(
        self,
        series_dict: Dict[str, pd.Series],
        calc_stride: Union[List[Union[float, pd.Timedelta]], None],
        segment_start_idxs: Union[np.ndarray, None],
        segment_end_idxs: Union[np.ndarray, None],
        start_idx: Any,
        end_idx: Any,
        window_idx: str,
        include_final_window: bool,
        approve_sparsity: bool,
    ) -> Callable[[int], Tuple[StridedRolling, FuncWrapper]]:
        # --- Future work ---
        # We could also make the StridedRolling creation multithreaded
        # Very low priority because the STROLL __init__ is rather efficient!
        keys_wins_strides = list(self._feature_desc_dict.keys())
        lengths = np.cumsum(
            [len(self._feature_desc_dict[k]) for k in keys_wins_strides]
        )

        def get_stroll_function(idx) -> Tuple[StridedRolling, FuncWrapper]:
            key_idx = np.searchsorted(lengths, idx, "right")  # right bc idx starts at 0
            key, win = keys_wins_strides[key_idx]

            feature = self._feature_desc_dict[keys_wins_strides[key_idx]][
                idx - lengths[key_idx]
            ]
            stride = feature.stride if calc_stride is None else calc_stride
            function: FuncWrapper = feature.function
            # The factory method will instantiate the right StridedRolling object
            stroll_arg_dict = dict(
                data=[series_dict[k] for k in key],
                window=win,
                strides=stride,
                segment_start_idxs=segment_start_idxs,
                segment_end_idxs=segment_end_idxs,
                start_idx=start_idx,
                end_idx=end_idx,
                window_idx=window_idx,
                include_final_window=include_final_window,
                approve_sparsity=approve_sparsity,
                func_data_type=function.input_type,
            )
            stroll = StridedRollingFactory.get_segmenter(**stroll_arg_dict)
            return stroll, function

        return get_stroll_function

    def _get_stroll_feat_length(self) -> int:
        return sum(
            len(self._feature_desc_dict[k]) for k in self._feature_desc_dict.keys()
        )

    def _check_no_multiple_windows(self):
        assert (
            self._get_nb_output_features_without_window()
            == self.get_nb_output_features()
        ), (
            "When using `segment_XXX_idxs`; each output name - series_input combination"
            + " can only have 1 window (or None)"
        )

    def _data_to_series_dict(
        self,
        data: Union[pd.DataFrame, pd.Series, List[Union[pd.Series, pd.DataFrame]]],
        required_series: List[str],
    ) -> Dict[str, pd.Series]:
        series_dict: Dict[str, pd.Series] = {}
        for s in to_series_list(data):
            if not s.index.is_monotonic_increasing:
                warnings.warn(
                    f"The index of series '{s.name}' is not monotonic increasing. "
                    + "The series will be sorted by the index.",
                    RuntimeWarning,
                )
                s = s.sort_index(ascending=True, inplace=False, ignore_index=False)

            # Assert the assumptions we make!
            assert s.index.is_monotonic_increasing

            if s.name in required_series:
                series_dict[str(s.name)] = s

        return series_dict

    @staticmethod
    def _process_segment_idxs(
        segment_idxs: Union[list, np.ndarray, pd.Series, pd.Index]
    ) -> np.ndarray:
        if hasattr(segment_idxs, "values"):
            segment_idxs = segment_idxs.values
        segment_idxs = np.asarray(segment_idxs)
        if segment_idxs.ndim > 1:
            segment_idxs = segment_idxs.squeeze()  # remove singleton dimensions
        return segment_idxs

    def _calculate_group_by(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        group_by: str,
        return_df: Optional[bool] = False,
        **calculate_kwargs,
    ):
        series_list = to_series_list(data)
        series_names = [s.name for s in series_list]
        # Check if the group_by column is a valid column
        assert (
            group_by in series_names
        ), f"Data contains no column named '{group_by}' to group by."
        # now extract all corresponding rows into separate series

        # get all unique values from group_by column
        group_by_unique_values = list(
            filter(lambda n: group_by == n.name, series_list)
        )[0].unique()

        # for convenience purposes, turn series_list into a DF
        df = pd.concat(series_list, axis=1)
        # now for every unique value, we can extract the corresponding rows
        extracted_dfs = []
        for unique_value in group_by_unique_values:
            unique_value_df = df.loc[df[group_by] == unique_value]
            extracted_dfs.append((unique_value, unique_value_df))

        # now for each different sub dataframe, recursively call calculate
        result_dfs = []
        for uv, extracted_df in extracted_dfs:
            try:
                # Todo: find a way to distribute available n_jobs
                warnings.filterwarnings("ignore", category=RuntimeWarning, message='^.*segment indexes.*$')
                calc_result = self.calculate(
                    data=extracted_df,
                    segment_start_idxs=[extracted_df.first_valid_index()],
                    segment_end_idxs=[extracted_df.last_valid_index() + 1],
                    **calculate_kwargs,
                )[0]

                result_dfs.append(calc_result.set_axis([uv], axis=0))
            except Exception as ex:
                warnings.warn(
                    f"An exception was raised during feature extraction:\n{ex}",
                    category=RuntimeWarning,
                )

        if return_df:
            # concatenate rows
            df = pd.concat(result_dfs, join="outer", copy=False)
            return df
        else:
            return result_dfs

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        stride: Optional[Union[float, str, pd.Timedelta, List, None]] = None,
        segment_start_idxs: Optional[
            Union[list, np.ndarray, pd.Series, pd.Index]
        ] = None,
        segment_end_idxs: Optional[Union[list, np.ndarray, pd.Series, pd.Index]] = None,
        return_df: Optional[bool] = False,
        window_idx: Optional[str] = "end",
        include_final_window: Optional[bool] = False,
        group_by: Optional[str] = None,
        bound_method: Optional[str] = "inner",
        approve_sparsity: Optional[bool] = False,
        show_progress: Optional[bool] = False,
        logging_file_path: Optional[Union[str, Path]] = None,
        n_jobs: Optional[int] = None,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Calculate features on the passed data.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            Dataframe or Series or list thereof, with all the required data for the
            feature calculation. \n
            **Assumptions**: \n
            * each Series / DataFrame must have a sortable index. This index represents
            the sequence position of the corresponding values, the index can be either
            numeric or a ``pd.DatetimeIndex``.
            * each Series / DataFrame index must be comparable with all others
            * we assume that each series-name / dataframe-column-name is unique.
        stride: Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta], None], optional
            The stride size. By default None. This argument supports multiple types: \n
            * If None, the stride of the `FeatureDescriptor` objects will be used.
            * If the type is an `float` or an `int`, its value represents the series:\n
                - its stride **range** when a **non time-indexed** series is passed.
                - the stride in **number of samples**, when a **time-indexed** series
                is passed (must then be and `int`)
            * If the stride's type is a `pd.Timedelta`, the stride size represents
            the stride-time delta. The passed data **must have a time-index**.
            * If a `str`, it must represent a stride-time-delta-string. Hence, the
            **passed data must have a time-index**. \n
            .. Note::
                When set, this stride argument takes precedence over the stride property
                of the `FeatureDescriptor`s in this `FeatureCollection` (i.e., when a
                not None value for `stride` passed to this method).
        segment_start_idxs: Union[list, np.ndarray, pd.Series, pd.Index], optional
            The start indices of the segments. If None, the start indices will be
            computed from the data using either:\n
            - the `segment_end_idxs` - the `window` size property of the
                `FeatureDescriptor` in this `FeatureCollection` (if `segment_end_idxs`
                is not None)
            - strided-window rolling on the data using `window` and `stride` of the
                `FeatureDescriptor` in this `FeatureCollection` (if `segment_end_idxs`
                 is also None). (Note that the `stride` argument of this method takes
                 precedence over the `stride` property of the `FeatureDescriptor`s).
            By default None.
        segment_end_idxs: Union[list, np.ndarray, pd.Series, pd.Index], optional
            The end indices for the segmented windows. If None, the end indices will be
            computed from the data using either:\n
            - the `segment_start_idxs` + the `window` size property of the
                `FeatureDescriptor` in this `FeatureCollection` (if `segment_start_idxs`
                is not None)
            - strided-window rolling on the data using `window` and `stride` of the
                `FeatureDescriptor` in this `FeatureCollection` (if `segment_start_idxs`
                 is also None). (Note that the `stride` argument of this method takes
                 precedence over the `stride` property of the `FeatureDescriptor`s).
            By default None.

            ..Note::
                When passing both `segment_start_idxs` and `segment_end_idxs`, these two
                arguments must have the same length and every start index must be <=
                than the corresponding end index.
                Note that passing both arguments, discards any meaning of the `window`
                and `stride` values (as these segment indices define the segmented data,
                and thus no strided-window rolling index calculation has to be executed).
                As such, the user can create variable-length segmented windows. However,
                in such cases, the user should be weary that the feature functions are
                invariant to these (potentially variable-length) windows.
        return_df : bool, optional
            Whether the output needs to be a DataFrame or a list thereof, by default
            False. If `True` the output dataframes will be merged to a DataFrame with an
            outer merge.
        window_idx : str, optional
            The window's index position which will be used as index for the
            feature_window aggregation. Must be either of: `["begin", "middle", "end"]`.
            by **default "end"**. All features in this collection will use the same
            window_idx.

            ..Note::
                `window_idx`="end" uses the window's end (= right open bound) as
                output index. \n
                `window_idx`="begin" uses the window's start idx (= left closed bound)
                as output index.
        include_final_window : bool, optional
            Whether the final (possibly incomplete) window should be included in the
            strided-window segmentation, by default False.

            .. Note::
                The remarks below apply when `include_final_window` is set to True.
                The user should be aware that the last window *might* be incomplete,
                i.e.;

                - when equally sampled, the last window *might* be smaller than the
                  the other windows.
                - when not equally sampled, the last window *might* not include all the
                  data points (as the begin-time + window-size comes after the last data
                  point).

                Note, that when equally sampled, the last window *will* be a full window
                when:

                - the stride is the sampling rate of the data (or stride = 1 for
                sample-based configurations).<br>
                **Remark**: that when `include_final_window` is set to False, the last
                window (which is a full) window will not be included!
                - *(len * sampling_rate - window_size) % stride = 0*. Remark that the
                  above case is a base case of this.
        group_by: str, optional
            The title of the column by which to perform grouping.
        bound_method: str, optional
            The start-end bound methodology which is used to generate the slice ranges
            when ``data`` consists of multiple series / columns.
            Must be either of: `["inner", "inner-outer", "outer"]`, by default "inner".

            * if ``inner``, the inner-bounds of the series are returned.
            * if ``inner-outer``, the left-inner and right-outer bounds of the series
              are returned.
            * if ``outer``, the outer-bounds of the series are returned.
        approve_sparsity: bool, optional
            Bool indicating whether the user acknowledges that there may be sparsity
            (i.e., irregularly sampled data), by default False.
            If False and sparsity is observed, a warning is raised.
        show_progress: bool, optional
            If True, the progress will be shown with a progressbar, by default False.
        logging_file_path : Union[str, Path], optional
            The file path where the logged messages are stored. If `None`, then no
            logging `FileHandler` will be used and the logging messages are only pushed
            to stdout. Otherwise, a logging `FileHandler` will write the logged messages
            to the given file path. See also the `tsflex.features.logger` module.
        n_jobs : int, optional
            The number of processes used for the feature calculation. If `None`, then
            the number returned by _os.cpu_count()_ is used, by default None. \n
            If n_jobs is either 0 or 1, the code will be executed sequentially without
            creating a process pool. This is very useful when debugging, as the stack
            trace will be more comprehensible.
            .. note::
                Multiprocessed execution is not supported on Windows. Even when,
                `n_jobs` is set > 1, the feature extraction will still be executed
                sequentially.
                Why do we not support multiprocessing on Windows; see this issue
                https://github.com/predict-idlab/tsflex/issues/51

            .. tip::
                It takes on avg. _300ms_ to schedule everything with
                multiprocessing. So if your sequential feature extraction code runs
                faster than ~1s, it might not be worth it to parallelize the process
                (and thus better leave `n_jobs` to 0 or 1).

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            The calculated features.

        Raises
        ------
        KeyError
            Raised when a required key is not found in `data`.

        Notes
        ------
        * The (column-)names of the series in `data` represent the `series_names`.
        * If a `logging_file_path` is provided, the execution (time) info can be
          retrieved by calling `logger.get_feature_logs(logging_file_path)`.
          Be aware that the `logging_file_path` gets cleared before the logger pushes
          logged messages. Hence, one should use a separate logging file for each
          constructed processing and feature instance with this library.


        """

        if group_by:
            return self._calculate_group_by(
                data,
                group_by,
                return_df,
                bound_method=bound_method,
                approve_sparsity=approve_sparsity,
                show_progress=show_progress,
                logging_file_path=logging_file_path,
                n_jobs=n_jobs,
            )

        # Delete other logging handlers
        delete_logging_handlers(logger)
        # Add logging handler (if path provided)
        if logging_file_path:
            f_handler = add_logging_handler(logger, logging_file_path)

        # Convert to numpy array (if necessary)
        if segment_start_idxs is not None:
            segment_start_idxs = FeatureCollection._process_segment_idxs(
                segment_start_idxs
            )
        if segment_end_idxs is not None:
            segment_end_idxs = FeatureCollection._process_segment_idxs(segment_end_idxs)

        if segment_start_idxs is not None and segment_end_idxs is not None:
            # Check if segment indices have same length and whether every start idx
            # <= end idx
            _check_start_end_array(segment_start_idxs, segment_end_idxs)
            # Check if there is either 1 or No(ne) window value for every output name -
            # input_series combination
            self._check_no_multiple_windows()

        if segment_start_idxs is None or segment_end_idxs is None:
            assert all(
                fd.window is not None
                for fd in flatten(self._feature_desc_dict.values())
            ), (
                "Each feature descriptor must have a window when not both "
                + "segment_start_idxs and segment_end_idxs are provided"
            )

        if stride is None and segment_start_idxs is None and segment_end_idxs is None:
            assert all(
                fd.stride is not None
                for fd in flatten(self._feature_desc_dict.values())
            ), (
                "Each feature descriptor must have a stride when no stride or "
                + "segment indices are passed to this method!"
            )
        elif stride is not None and (
            segment_start_idxs is not None or segment_end_idxs is not None
        ):
            raise ValueError(
                "The stride and any segment index argument cannot be set together! "
                + "At least one of both should be None."
            )

        if stride is not None:
            # Verify whether the stride complies with the input data dtype
            stride = [
                parse_time_arg(s) if isinstance(s, str) else s for s in to_list(stride)
            ]
            self._check_feature_descriptors(skip_none=False, calc_stride=stride)

        # Convert the data to a series_dict
        series_dict = self._data_to_series_dict(data, self.get_required_series())

        # Determine the bounds of the series dict items and slice on them
        # TODO: is dit wel nodig `hier? want we doen dat ook in de strided rolling
        start, end = _determine_bounds(bound_method, list(series_dict.values()))
        series_dict = {
            n: s.loc[
                s.index.dtype.type(start) : s.index.dtype.type(end)
            ]  # TODO: check memory efficiency of ths
            for n, s, in series_dict.items()
        }

        # Note: this variable has a global scope so this is shared in multiprocessing
        # TODO: try to make this more efficient (but is not really the bottleneck)
        global get_stroll_func
        get_stroll_func = self._stroll_feat_generator(
            series_dict,
            calc_stride=stride,
            segment_start_idxs=segment_start_idxs,
            segment_end_idxs=segment_end_idxs,
            start_idx=start,
            end_idx=end,
            window_idx=window_idx,
            include_final_window=include_final_window,
            approve_sparsity=approve_sparsity,
        )
        nb_stroll_funcs = self._get_stroll_feat_length()

        if (
            os.name == "nt"
        ):  # On Windows no multiprocessing is supported, see https://github.com/predict-idlab/tsflex/issues/51
            n_jobs = 1
        elif n_jobs is None:
            n_jobs = os.cpu_count()
        n_jobs = min(n_jobs, nb_stroll_funcs)

        calculated_feature_list = None
        if n_jobs in [0, 1]:
            idxs = range(nb_stroll_funcs)
            if show_progress:
                idxs = tqdm(idxs)
            try:
                calculated_feature_list = [self._executor(idx) for idx in idxs]
            except Exception:
                traceback.print_exc()
        else:
            with Pool(processes=n_jobs) as pool:
                results = pool.imap_unordered(self._executor, range(nb_stroll_funcs))
                if show_progress:
                    results = tqdm(results, total=nb_stroll_funcs)
                try:
                    calculated_feature_list = [f for f in results]
                except Exception:
                    traceback.print_exc()
                    pool.terminate()
                finally:
                    # Close & join because: https://github.com/uqfoundation/pathos/issues/131
                    pool.close()
                    pool.join()

        # Close the file handler (this avoids PermissionError: [WinError 32])
        if logging_file_path:
            f_handler.close()
            logger.removeHandler(f_handler)

        if calculated_feature_list is None:
            raise RuntimeError(
                "Feature Extraction halted due to error while extracting one "
                + "(or multiple) feature(s)! See stack trace above."
            )

        if return_df:
            # concatenate & sort the columns
            df = pd.concat(calculated_feature_list, axis=1, join="outer", copy=False)
            return df.reindex(sorted(df.columns), axis=1)
        else:
            return calculated_feature_list

    def serialize(self, file_path: Union[str, Path]):
        """Serialize this FeatureCollection instance.

        Parameters
        ----------
        file_path : Union[str, Path]
            The path where the `FeatureCollection` will be serialized.

        Note
        -----
        As we use [Dill](https://github.com/uqfoundation/dill){:target="_blank"} to
        serialize the files, we can **also serialize functions which are defined in
        the local scope, like lambdas.**

        """
        with open(file_path, "wb") as f:
            dill.dump(self, f, recurse=True)

    def reduce(self, feat_cols_to_keep: List[str]) -> FeatureCollection:
        """Create a reduced FeatureCollection instance based on `feat_cols_to_keep`.

        For example, this is useful to optimize feature-extraction inference
        (for your selected features) after performing a feature-selection procedure.

        Parameters
        ----------
        feat_cols_to_keep: List[str]
            A subset of the feature collection instance its column names.
            This corresponds to the columns / names of the output from `calculate`
            method that you want to keep.

        Returns
        -------
        FeatureCollection
            A new FeatureCollection object, which only withholds the FeatureDescriptors
            which constitute the `feat_cols_to_keep` output.

        Note
        ----
        Some FeatureDescriptor objects may have multiple **output-names**.<br>
        Hence, if you only want to retain _a subset_ of that FeatureDescriptor its
        feature outputs, you will still get **all features** as the new
        FeatureCollection is constructed by applying a filter on de FeatureDescriptor
        list and we thus not alter these FeatureDescriptor objects themselves.

        """
        # dict in which we store all the { output_col_name : (UUID, FeatureDescriptor) }
        # items of our current FeatureCollection object
        manual_window = False
        if any(c.endswith("w=manual") for c in feat_cols_to_keep):
            assert all(c.endswith("w=manual") for c in feat_cols_to_keep)
            # As the windows are created manual, the FeatureCollection cannot contain
            # multiple windows for the same output name - input_series combination
            self._check_no_multiple_windows()
            manual_window = True
        feat_col_fd_mapping: Dict[str, Tuple[str, FeatureDescriptor]] = {}
        for (s_names, window), fd_list in self._feature_desc_dict.items():
            window = "manual" if manual_window else self._ws_to_str(window)
            for fd in fd_list:
                # As a single FeatureDescriptor can have multiple output col names, we
                # create a unique identifier for each FeatureDescriptor (on which we
                # will apply set-like operations later on to only retain all the unique
                # FeatureDescriptors)
                uuid_str = str(uuid.uuid4())
                for output_name in fd.function.output_names:
                    # Reconstruct the feature column name
                    feat_col_name = StridedRolling.construct_output_index(
                        series_keys=s_names, feat_name=output_name, win_str=window
                    )
                    feat_col_fd_mapping[feat_col_name] = (uuid_str, fd)

        assert all(fc in feat_col_fd_mapping for fc in feat_cols_to_keep)

        # Collect (uuid, FeatureDescriptor) for the feat_cols_to_keep
        fd_subset: List[Tuple[str, FeatureDescriptor]] = [
            feat_col_fd_mapping[fc] for fc in feat_cols_to_keep
        ]

        # Reduce to unique feature descriptor objects (based on uuid) and create a new
        # FeatureCollection for their deepcopy's.
        seen_uuids = set()
        return FeatureCollection(
            feature_descriptors=[
                deepcopy(unique_fd)
                for unique_fd in {
                    fd
                    for (uuid_str, fd) in fd_subset
                    if uuid_str not in seen_uuids and not seen_uuids.add(uuid_str)
                }
            ]
        )

    @staticmethod
    def _ws_to_str(window_or_stride: Any) -> str:
        """Convert the window/stride value to a (shortend) string representation."""
        if isinstance(window_or_stride, pd.Timedelta):
            return timedelta_to_str(window_or_stride)
        else:
            return str(window_or_stride)

    def __repr__(self) -> str:
        """Representation string of a FeatureCollection."""
        feature_keys = sorted(set(k[0] for k in self._feature_desc_dict.keys()))
        output_str = ""
        for feature_key in feature_keys:
            output_str += f"{'|'.join(feature_key)}: ("
            keys = (x for x in self._feature_desc_dict.keys() if x[0] == feature_key)
            for _, win_size in keys:
                output_str += "\n\twin: "
                win_str = self._ws_to_str(win_size)
                output_str += f"{win_str:<6}: ["
                for feat_desc in self._feature_desc_dict[feature_key, win_size]:
                    stride_str = feat_desc.stride
                    if stride_str is not None:
                        stride_str = [self._ws_to_str(s) for s in stride_str]
                    output_str += f"\n\t\t{feat_desc._func_str}"
                    output_str += f"    stride: {stride_str},"
                output_str += "\n\t]"
            output_str += "\n)\n"
        return output_str
