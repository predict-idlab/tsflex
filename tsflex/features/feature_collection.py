"""FeatureCollection class for bookkeeping and calculation of time-series features.

Methods, next to `FeatureCollection.calculate()`, worth looking at: \n
* `FeatureCollection.serialize()` - serialize the FeatureCollection to a file
* `FeatureCollection.reduce()` - reduce the number of features after feature selection

"""

from __future__ import annotations
import warnings


__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import dill
import traceback
import numpy as np
import pandas as pd
from multiprocess import Pool
from tqdm.auto import tqdm

from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .logger import logger
from .segmenter import StridedRollingFactory, StridedRolling
from .utils import _determine_bounds
from ..features.function_wrapper import FuncWrapper
from ..utils.attribute_parsing import AttributeParser
from ..utils.data import to_list, to_series_list, flatten
from ..utils.logging import delete_logging_handlers, add_logging_handler
from ..utils.time import timedelta_to_str

if os.name == "nt":  # If running on Windows
    # This enables pickling of globals on Windows
    dill.settings["recurse"] = True
    dill.settings["byref"] = True


class FeatureCollection:
    """Create a FeatureCollection.

    Parameters
    ----------
    feature_descriptors : Union[FeatureDescriptor, MultipleFeatureDescriptors, List[Union[FeatureDescriptor, MultipleFeatureDescriptors]]], optional
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
                List[Union[FeatureDescriptor, MultipleFeatureDescriptors]],
            ]
        ] = None,
    ):
        # The feature collection is a dict with keys of type:
        #   tuple(tuple(str), float OR pd.timedelta, float OR pd.timedelta)
        # The outer tuple's values correspond to (series_key(s), window, stride)
        self._feature_desc_dict: Dict[
            Tuple[
                Tuple[str, ...], Union[float, pd.Timedelta], Union[float, pd.Timedelta]
            ],
            List[FeatureDescriptor],
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

    @staticmethod
    def _get_collection_key(
        feature: FeatureDescriptor,
    ) -> Tuple[tuple, Union[pd.Timedelta, float], Union[pd.Timedelta, float]]:
        # Note: `window` & `stride` properties can either be a pd.Timedelta or an int
        return feature.series_name, feature.window, feature.stride

    def _check_feature_descriptors(self):
        """Verify whether all added FeatureDescriptors imply the same-input data type.

        If this condition is not met, a warning will be raised.

        """
        dtype_set = set(
            AttributeParser.determine_type([win, stride])
            for _, win, stride in self._feature_desc_dict.keys()
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
        self._check_feature_descriptors()

    @staticmethod
    def _executor(idx: int):
        # global get_stroll_func
        stroll, function = get_stroll_func(idx)
        return stroll.apply_func(function)

    def _stroll_feat_generator(
        self,
        series_dict: Dict[str, pd.Series],
        start_idx: Any,
        end_idx: Any,
        window_idx: str,
        approve_sparsity: bool,
    ) -> List[Tuple[StridedRolling, FuncWrapper]]:
        # --- Future work ---
        # We could also make the StridedRolling creation multithreaded
        # Very low priority because the STROLL __init__ is rather efficient!
        keys_wins_strides = list(self._feature_desc_dict.keys())
        lengths = np.cumsum(
            [len(self._feature_desc_dict[k]) for k in keys_wins_strides]
        )

        def get_stroll_function(idx):
            key_idx = np.searchsorted(lengths, idx, "right")  # right bc idx starts at 0
            key, win, stride = keys_wins_strides[key_idx]
            feature = self._feature_desc_dict[keys_wins_strides[key_idx]][
                idx - lengths[key_idx]
            ]
            function: FuncWrapper = feature.function
            # The factory method will instantiate the right StridedRolling object
            stroll_arg_dict = dict(
                data=[series_dict[k] for k in key],
                window=win,
                stride=stride,
                start_idx=start_idx,
                end_idx=end_idx,
                window_idx=window_idx,
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

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        return_df: Optional[bool] = False,
        window_idx: Optional[str] = "end",
        bound_method: Optional[str] = "inner",
        approve_sparsity: Optional[bool] = False,
        show_progress: Optional[bool] = False,
        logging_file_path: Optional[Union[str, Path]] = None,
        n_jobs: Optional[int] = None,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Calculate features on the passed data.

        Notes
        ------
        * The (column-)names of the series in `data` represent the `series_names`.
        * If a `logging_file_path` is provided, the execution (time) info can be
          retrieved by calling `logger.get_feature_logs(logging_file_path)`.
          Be aware that the `logging_file_path` gets cleared before the logger pushes
          logged messages. Hence, one should use a separate logging file for each
          constructed processing and feature instance with this library.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            Dataframe or Series or list thereof, with all the required data for the
            feature calculation. \n
            **Assumptions**: \n
            * each Series / DataFrame must have a sortable index. This index represents 
            the sequence position of the corresponding values, the index can be either 
            numeric or a ``pd.DatetimeIndex``.
            * each Series / DataFrame index must be comparable.with all others
            * we assume that each series-name / dataframe-column-name is unique.
        return_df : bool, optional
            Whether the output needs to be a DataFrame or a list thereof, by default
            False. If `True` the output dataframes will be merged to a DataFrame with an 
            outer merge.
        window_idx : str, optional
            The window's index position which will be used as index for the
            feature_window aggregation. Must be either of: `["begin", "middle", "end"]`.
            by default "end". All features in this collection will use the same 
            window_idx.
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

            .. tip::
                It takes on avg. _300ms_ to schedule everything with
                multiprocessing. So if your sequential feature extraction code runs
                faster than ~1.5s, it might not be worth it to parallelize the process
                (and thus better leave `n_jobs` to 0 or 1).

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            The calculated features.

        Raises
        ------
        KeyError
            Raised when a required key is not found in `data`.

        """
        # Delete other logging handlers
        delete_logging_handlers(logger)
        # Add logging handler (if path provided)
        if logging_file_path:
            f_handler = add_logging_handler(logger, logging_file_path)

        # Convert the data to a series_dict
        series_dict: Dict[str, pd.Series] = {}
        for s in to_series_list(data):
            if not s.index.is_monotonic_increasing:
                # TODO -> maybe raise a warning?
                s = s.sort_index(ascending=True, inplace=False, ignore_index=False)

            # Assert the assumptions we make!
            assert s.index.is_monotonic_increasing

            if s.name in self.get_required_series():
                series_dict[str(s.name)] = s

        # determing the bounds of the series dict items and slice on them
        start, end = _determine_bounds(bound_method, list(series_dict.values()))
        series_dict = {
            n: s[s.index.dtype.type(start) : s.index.dtype.type(end)]
            for n, s, in series_dict.items()
        }

        # Note: this variable has a global scope so this is shared in multiprocessing
        # TODO: try to make this more efficient
        global get_stroll_func
        get_stroll_func = self._stroll_feat_generator(
            series_dict, start, end, window_idx, approve_sparsity
        )
        nb_stroll_funcs = self._get_stroll_feat_length()

        if n_jobs is None:
            n_jobs = os.cpu_count()
        n_jobs = min(n_jobs, nb_stroll_funcs)

        calculated_feature_list = None
        if n_jobs in [0, 1]:
            idxs = range(nb_stroll_funcs)
            if show_progress:
                idxs = tqdm(idxs)
            try:
                calculated_feature_list = [self._executor(idx) for idx in idxs]
            except:
                traceback.print_exc()
        else:
            with Pool(processes=n_jobs) as pool:
                chunk_size = 100 if os.name == "nt" else 1
                results = pool.imap_unordered(self._executor, range(nb_stroll_funcs), chunk_size)
                if show_progress:
                    results = tqdm(results, total=nb_stroll_funcs)
                try:
                    calculated_feature_list = [f for f in results]
                except:
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
                "Feature Extraction halted due to error while extracting one (or multiple) feature(s)! "
                + "See stack trace above."
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
        # items of our current Featurecollection object
        feat_col_fd_mapping: Dict[str, Tuple[str, FeatureDescriptor]] = {}
        for (s_names, window, stride), fds in self._feature_desc_dict.items():
            fd: FeatureDescriptor
            for fd in fds:
                # As a single FeatureDescriptor can have multiple output col names, we
                # create a unique identifier for each FeatureDescriptor (on which we
                # will apply set-like operations later on to only retain all the unique
                # FeatureDescriptors)
                uuid_str = str(uuid.uuid4())
                for output_name in fd.function.output_names:
                    # Reconstruct the feature column name
                    feat_col_name = "__".join(
                        [
                            "|".join(s_names)
                            if isinstance(s_names, tuple)
                            else s_names,
                            output_name,
                            f"w={self._ws_to_str(window)}_s={self._ws_to_str(stride)}",
                        ]
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
            for _, win_size, stride in keys:
                output_str += f"\n\twin: "
                win_str = self._ws_to_str(win_size)
                stride_str = self._ws_to_str(stride)
                output_str += f"{win_str:<6}, stride: {stride_str}: ["
                for feat_desc in self._feature_desc_dict[feature_key, win_size, stride]:
                    output_str += f"\n\t\t{feat_desc._func_str},"
                output_str += "\n\t]"
            output_str += "\n)\n"
        return output_str
