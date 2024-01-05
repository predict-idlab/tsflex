"""FeatureCollection class for bookkeeping and calculation of time-series features.

Methods, next to `FeatureCollection.calculate()`, worth looking at: \n
* `FeatureCollection.serialize()` - serialize the FeatureCollection to a file
* `FeatureCollection.reduce()` - reduce the number of features after feature selection

"""

from __future__ import annotations

import warnings

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

import logging
import os
import time
import traceback
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import dill
import numpy as np
import pandas as pd
from multiprocess import Pool
from pandas.api.types import is_datetime64_any_dtype
from tqdm.auto import tqdm

from ..features.function_wrapper import FuncWrapper
from ..utils.attribute_parsing import AttributeParser
from ..utils.data import flatten, to_list, to_series_list
from ..utils.logging import add_logging_handler, delete_logging_handlers
from ..utils.time import parse_time_arg, timedelta_to_str
from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .logger import logger
from .segmenter import StridedRolling, StridedRollingFactory
from .utils import (
    _check_start_end_array,
    _determine_bounds,
    _log_func_execution,
    _process_func_output,
)


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

    def _get_nb_feat_funcs(self) -> int:
        return sum(
            len(self._feature_desc_dict[k]) for k in self._feature_desc_dict.keys()
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
    def _executor_stroll(idx: int) -> pd.DataFrame:
        """Executor function for the StridedRolling.apply_func method.

        Strided rolling feature calculation occurs when either;
        - a `window` and `stride` argument are stored in the `FeatureDescriptor` object
        - the `window` is stored in the `FeatureDescriptor` object and the `stride`
          argument is passed to the `calculate` method, potentially overriding the
          `stride`
        - segment indices are passed to the `calculate` method
        - a `group_by_consecutive` argument is passed to the `calculate` method (since
          we calculate the segment indices for the consecutive groups)

        This method uses the global `get_stroll_func` function, which returns the
        StridedRolling object and the function that needs to be applied to the
        StridedRolling object. Using a global function is necessary to facilitate
        multiprocessing.
        """
        # Uses the global get_stroll_func
        stroll, function = get_stroll_func(idx)
        return stroll.apply_func(function)  # execution time is logged in apply_func

    @staticmethod
    def _executor_grouped(idx: int) -> pd.DataFrame:
        """Executor function for grouped feature calculation.

        Grouped feature calculation occurs when either;
        - a `group_by_all` argument is passed to the `calculate` method
        - a `DataFrameGroupBy` is passed as `data` argument to the `calculate` method

        Note that passing a `group_by_consecutive` argument to the `calculate` method
        will not use this executor function, but will use the `_executor_stroll` as
        executor function (since we calculate the segment indices for the consecutive
        groups).

        This method uses the global `get_group_func` function, which returns a
        pd.DataFrame (containing only the necessary data for the function) and the
        function that needs to be applied to the pd.DataFrame. In addition, the global
        `group_indices` and `group_id_name` are used to access the grouped data and the
        group id name respectively. Using a global function is necessary to facilitate
        multiprocessing.
        """
        # Uses the global get_group_func, group_indices, and group_id_name
        data, function = get_group_func(idx)
        group_ids = group_indices.keys()  # group_ids are the keys of the group_indices
        cols = data.columns.values

        t_start = time.perf_counter()

        # Wrap the function to handle multiple inputs and convert the inputs to numpy
        # array if necessary
        f = function
        if function.input_type is np.array:

            def f(x: pd.DataFrame):
                # pass the inputs as positional arguments of numpy array type
                return function(*[x[c].values for c in cols])

        else:  # function.input_type is pd.Series

            def f(x: pd.DataFrame):
                # pass the inputs as positional arguments of pd.Series type
                return function(*[x[c] for c in cols])

        # Function execution over the grouped data (accessed by using the group_indices)
        out = np.array(list(map(f, [data.iloc[idx] for idx in group_indices.values()])))

        # Aggregate function output in a dictionary
        output_names = [
            StridedRolling.construct_output_index(cols, feat_name, win_str="manual")
            for feat_name in function.output_names
        ]
        feat_out = _process_func_output(out, group_ids, output_names, str(function))

        # Log the function execution time
        _log_func_execution(
            t_start, function, tuple(cols), "manual", "manual", output_names
        )

        return pd.DataFrame(feat_out, index=group_ids).rename_axis(index=group_id_name)

    def _group_feat_generator(
        self,
        grouped_df: pd.api.typing.DataFrameGroupBy,
    ) -> Callable[[int], Tuple[pd.api.typing.DataFrameGroupBy, FuncWrapper,],]:
        keys_wins = list(self._feature_desc_dict.keys())
        lengths = np.cumsum([len(self._feature_desc_dict[k]) for k in keys_wins])

        def get_group_function(
            idx,
        ) -> Tuple[pd.api.typing.DataFrameGroupBy, FuncWrapper,]:
            key_idx = np.searchsorted(lengths, idx, "right")  # right bc idx starts at 0
            key, win = keys_wins[key_idx]

            feature = self._feature_desc_dict[keys_wins[key_idx]][
                idx - lengths[key_idx]
            ]
            function: FuncWrapper = feature.function
            return grouped_df.obj[list(key)], function

        return get_group_function

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

    def _check_no_multiple_windows(self, error_case: str):
        """Check whether there are no multiple windows in the feature collection.

        Parameters
        ----------
        error_case : str
            The case in which no multiple windows are allowed.

        """
        assert (
            self._get_nb_output_features_without_window()
            == self.get_nb_output_features()
        ), (
            error_case
            + "; each output name - series_input combination can only have 1 window"
            + " (or None)"
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

    @staticmethod
    def _group_by_all(
        series_dict: Dict[str, pd.Series], col_name: str = None
    ) -> pd.api.typing.DataFrameGroupBy:
        """Group all `column_name` values and return the grouped data.

        Parameters
        ----------
        series_dict : Dict[str, pd.Series]
            Input data.
        col_name : str
            The column name on which the grouping will need to take place.

        Returns
        -------
        pd.api.typing.DataFrameGroupBy
            A `DataFrameGroupBy` object, with the group names as keys and the indices
            as values.

        """
        df = pd.DataFrame(series_dict)
        assert col_name in df.columns

        # GroupBy ignores all rows with NaN values for the column on which we group
        return df.groupby(col_name)

    def _calculate_group_by_all(
        self,
        grouped_data: pd.api.typing.DataFrameGroupBy,
        return_df: Optional[bool],
        show_progress: Optional[bool],
        n_jobs: Optional[int],
        f_handler: Optional[logging.FileHandler],
    ):
        """Calculate features on each group of the grouped data.

        Parameters
        ----------
        grouped_data : pd.api.typing.DataFrameGroupBy
            The grouped data.
        return_df: bool, optional
            Whether the output needs to be a DataFrame or a list thereof.
        show_progress: bool, optional
            Whether to show a progress bar.
        n_jobs: int, optional
            The number of jobs to run in parallel.
        f_handler: logging.FileHandler, optional
            The file handler that is used to log the function execution times.

        .. Note::
            Is comparable to following pseudo-SQL code:
            ```sql
            SELECT func(x)
            FROM `data`
            GROUP BY ...
            ```
            where `func` is the FeatureDescriptor function and `x` is the name
            on which the FeatureDescriptor operates. The group by is already done by
            passing a `DataFrameGroupBy` object to this method.
        """
        global group_indices, group_id_name, get_group_func
        group_indices = grouped_data.indices  # dict - group_id as key; indices as value
        group_id_name = grouped_data.grouper.names  # name of the group col(s)
        get_group_func = self._group_feat_generator(grouped_data)

        return self._calculate_feature_list(
            self._executor_grouped, n_jobs, show_progress, return_df, f_handler
        )

    @staticmethod
    def _group_by_consecutive(
        df: Union[pd.Series, pd.DataFrame], col_name: str = None
    ) -> pd.DataFrame:
        """Group consecutive `col_name` values in a single DataFrame.

        This is especially useful if you want to represent sparse data in a more
        compact format.

        Parameters
        ----------
        df : Union[pd.Series, pd.DataFrame]
            Input data.
        col_name : str, optional
            If a dataFrame is passed, you will need to specify the `col_name` on which
            the consecutive-grouping will need to take place.

        Returns
        -------
        pd.DataFrame
            A new `DataFrame` view, with columns:
            [`start`, `end`, `col_name`], representing the
            start- and endtime of the consecutive range, and the col_name's consecutive
            values.
        """
        if type(df) == pd.Series:
            col_name = df.name
            df = df.to_frame()

        assert col_name in df.columns

        # Drop all rows with NaN values for the column on which we group
        df.dropna(subset=[col_name], inplace=True)

        df_cum = (
            (df[col_name] != df[col_name].shift(1))
            .astype("int")
            .cumsum()
            .rename("value_grp")
            .to_frame()
        )
        df_cum["sequence_idx"] = df.index
        df_cum[col_name] = df[col_name]

        df_cum_grouped = df_cum.groupby("value_grp")
        df_grouped = pd.DataFrame(
            {
                "start": df_cum_grouped["sequence_idx"].first(),
                "end": df_cum_grouped["sequence_idx"].last(),
                col_name: df_cum_grouped[col_name].first(),
            }
        ).reset_index(drop=True)

        return df_grouped

    def _calculate_group_by_consecutive(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        group_by: str,
        return_df: Optional[bool] = False,
        **calculate_kwargs,
    ):
        """Calculate features on each consecutive group of the data.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            Must be time-indexed!
        group_by: str
            Name of column by which to group values.
        return_df: bool, optional
            Whether the output needs to be a DataFrame or a list thereof, by default
            False. If `True` the output dataframes will be merged to a DataFrame with an
            outer merge.
        **calculate_kwargs:
            Keyword arguments that will be passed to the `calculate` method.

        .. Note::
            Is comparable to following pseudo-SQL code:
            ```sql
            SELECT func(x)
            FROM `data`
            GROUP BY `group_by`
            ```
            where `func` is the FeatureDescriptor function and `x` is the name
            on which the FeatureDescriptor operates. Note however that the grouping is
            done on consecutive values of `group_by` (i.e. `group_by` values that are
            the same and are next to each other are grouped together).
        """
        # 0. Transform to dataframe
        series_dict = self._data_to_series_dict(
            data, self.get_required_series() + [group_by]
        )
        df = pd.DataFrame(series_dict)
        # 1. Group by `group_by` column
        consecutive_grouped_by_df = self._group_by_consecutive(df, col_name=group_by)
        # 2. Get start and end idxs of consecutive groups
        start_segment_idxs = consecutive_grouped_by_df["start"]
        end_segment_idxs = start_segment_idxs.shift(-1).fillna(
            consecutive_grouped_by_df["end"]
        )
        # because segment end idxs are exclusive, we need to add an offset to the last
        # end idx so that all data gets used
        segment_vals = end_segment_idxs.values
        if is_datetime64_any_dtype(segment_vals):
            segment_vals[-1] += pd.Timedelta(days=1)
        else:
            segment_vals[-1] += 1
        # 3. Calculate features
        try:
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="^.*segment indexes.*$"
            )
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="^.*gaps.*$"
            )
            # 3.1. Calculate features using the groupby segment idxs
            calc_results = self.calculate(
                data=df,
                segment_start_idxs=start_segment_idxs,
                segment_end_idxs=end_segment_idxs,
                **calculate_kwargs,
            )

            warnings.resetwarnings()

            # 3.2 Concatenate results and add the group_by column as well as the
            # start and end idxs of the segments
            calc_result = pd.concat(calc_results, join="outer", copy=False, axis=1)
            calc_result.reset_index(inplace=True, drop=True)
            calc_result[group_by] = consecutive_grouped_by_df[group_by]
            calc_result["__start"] = consecutive_grouped_by_df["start"]
            calc_result["__end"] = consecutive_grouped_by_df["end"]

            if return_df:
                # concatenate rows
                return calc_result
            else:
                return [calc_result[col] for col in calc_result.columns]

        except Exception as e:
            raise RuntimeError(
                f"An exception was raised during feature extraction:\n{e}"
            )

    @staticmethod
    def _process_njobs(n_jobs: Union[int, None], nb_funcs: int) -> int:
        """Process the number of jobs to run in parallel.

        On Windows no multiprocessing is supported, see
        https://github.com/predict-idlab/tsflex/issues/51

        Parameters
        ----------
        n_jobs : Union[int, None]
            The number of jobs to run in parallel.
        nb_funcs : int
            The number of feature functions.

        Returns
        -------
        int
            The number of jobs to run in parallel.

        """
        if os.name == "nt":  # On Windows no multiprocessing is supported
            n_jobs = 1
        elif n_jobs is None:
            n_jobs = os.cpu_count()
        return min(n_jobs, nb_funcs)

    def _calculate_feature_list(
        self,
        executor: Callable[[int], pd.DataFrame],
        n_jobs: Optional[int],
        show_progress: Optional[bool],
        return_df: Optional[bool],
        f_handler: Optional[logging.FileHandler],
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Calculate the features for the given executor.

        Parameters
        ----------
        executor : Callable[[int], pd.DataFrame]
            The executor function that will be used to calculate the features.
        n_jobs : Optional[int], optional
            The number of jobs to run in parallel.
        show_progress : Optional[bool], optional
            Whether to show a progress bar.
        return_df : Optional[bool], optional
            Whether to return a DataFrame or a list of DataFrames.
        f_handler : Optional[logging.FileHandler], optional
            The file handler that is used to log the function execution times.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            The calculated features.

        """
        nb_feat_funcs = self._get_nb_feat_funcs()
        n_jobs = FeatureCollection._process_njobs(n_jobs, nb_feat_funcs)

        calculated_feature_list: List[pd.DataFrame] = None

        if n_jobs in [0, 1]:
            # No multiprocessing
            idxs = range(nb_feat_funcs)
            if show_progress:
                idxs = tqdm(idxs)
            try:
                calculated_feature_list = [executor(idx) for idx in idxs]
            except Exception:
                traceback.print_exc()
        else:
            # Multiprocessing
            with Pool(processes=n_jobs) as pool:
                results = pool.imap_unordered(executor, range(nb_feat_funcs))
                if show_progress:
                    results = tqdm(results, total=nb_feat_funcs)
                try:
                    calculated_feature_list = [f for f in results]
                except Exception:
                    traceback.print_exc()
                    pool.terminate()
                finally:
                    # Close & join because: https://github.com/uqfoundation/pathos/issues/131
                    pool.close()
                    pool.terminate()
                    pool.join()

        # Close the file handler (this avoids PermissionError: [WinError 32])
        if f_handler is not None:
            f_handler.close()
            logger.removeHandler(f_handler)

        if calculated_feature_list is None:
            raise RuntimeError(
                "Feature Extraction halted due to error while extracting one "
                + "(or multiple) feature(s)! See stack trace above."
            )

        if return_df:
            # Concatenate & sort the columns
            df = pd.concat(calculated_feature_list, axis=1, join="outer", copy=False)
            return df.reindex(sorted(df.columns), axis=1)
        else:
            return calculated_feature_list

    def calculate(
        self,
        data: Union[
            pd.Series,
            pd.DataFrame,
            List[Union[pd.Series, pd.DataFrame]],
            pd.core.groupby.DataFrameGroupby,
        ],
        stride: Optional[Union[float, str, pd.Timedelta, List, None]] = None,
        segment_start_idxs: Optional[
            Union[list, np.ndarray, pd.Series, pd.Index]
        ] = None,
        segment_end_idxs: Optional[Union[list, np.ndarray, pd.Series, pd.Index]] = None,
        return_df: Optional[bool] = False,
        window_idx: Optional[str] = "end",
        include_final_window: Optional[bool] = False,
        group_by_all: Optional[str] = None,  # TODO: support multiple columns
        group_by_consecutive: Optional[str] = None,  # TODO: support multiple columns
        bound_method: Optional[str] = "inner",
        approve_sparsity: Optional[bool] = False,
        show_progress: Optional[bool] = False,
        logging_file_path: Optional[Union[str, Path]] = None,
        n_jobs: Optional[int] = None,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Calculate features on the passed data.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]], pd.core.groupby.DataFrameGroupby]
            Dataframe or Series or list thereof, with all the required data for the
            feature calculation. \n
            **Assumptions**: \n
            * each Series / DataFrame must have a sortable index. This index represents
            the sequence position of the corresponding values, the index can be either
            numeric or a ``pd.DatetimeIndex``.
            * each Series / DataFrame index must be comparable with all others
            * we assume that each series-name / dataframe-column-name is unique.
            Can also be a `DataFrameGroupBy` object, in which case the expected
            behaviour is similar to grouping by all values in `group_by_all` (i.e.,
            for each group, the features are calculated on the group's data).
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
        group_by_all : str, optional
            The name of the column by which to perform grouping. For each group, the
            features will be calculated. The output that is returned contains this
            `group_by` column as index to allow identifying the groups.
            If this parameter is used, the parameters `stride`, `segment_start_idxs`,
            `segment_end_idxs`, `window_idx` and `include_final_window` will be ignored.
            Rows with NaN values for this column will not be considered (as pandas
            ignores these rows when grouping).
            .. note::
                This is similar as passing a `DataFrameGroupBy` object as `data`
                argument to the `calculate` method, where the `DataFrameGroupBy` object
                is created by calling `data.groupby(group_by_all)`.
        group_by_consecutive: str, optional
            The name of the column by which to perform consecutive grouping. A
            consecutive group is a group of values that are the same and are next to
            each other. For each consecutive group, the features will be calculated.
            The output that is returned contains this `group_by` column to allow
            identifying the groups, and also contains fields [`__start`, "__end"] which
            contain start and end time range for each result row.
            If this parameter is used, the parameters `stride`, `segment_start_idxs`,
            `segment_end_idxs`, `window_idx` and `include_final_window` will be ignored.
            Rows with NaN values for this column will not be considered (as we deem NaN
            not as a groupable value).
            Note that for consecutive grouping, groups can appear multiple times if they
            appear in different time-gaps.

            Example output:
            .. example::
        ```python
                    number_sold__sum__w=manual  store    __start      __end
                0          845                  0     2019-01-01 2019-01-01
                1          357                  3     2019-01-02 2019-01-02
                2          904                  6     2019-01-03 2019-01-03
                3          599                  3     2019-01-04 2019-01-05
                4          871                  0     2019-01-06 2019-01-06
                ...                           ...    ...        ...        ...
        ```
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

        # Check valid data
        if isinstance(data, list):
            assert all(
                isinstance(d, (pd.Series, pd.DataFrame)) for d in data
            ), "All elements of the data list must be either a Series or a DataFrame!"
        else:
            assert isinstance(
                data, (pd.Series, pd.DataFrame, pd.core.groupby.DataFrameGroupBy)
            ), "The data must be either a Series, a DataFrame or a DataFrameGroupBy!"

        # check valid group_by
        assert group_by_all is None or group_by_consecutive is None, (
            "Only max one of the following parameters can be set: "
            + "`group_by_all` or `group_by_consecutive`"
        )
        assert not (
            (group_by_all is not None or group_by_consecutive is not None)
            and isinstance(data, pd.core.groupby.DataFrameGroupBy)
        ), (
            "Cannot use `group_by_all` or `group_by_consecutive` when `data` is"
            + " already a grouped DataFrame!"
        )

        # Delete other logging handlers
        delete_logging_handlers(logger)
        # Add logging handler (if path provided)
        f_handler = None
        if logging_file_path:
            f_handler = add_logging_handler(logger, logging_file_path)

        if (
            group_by_all
            or group_by_consecutive
            or isinstance(data, pd.core.groupby.DataFrameGroupBy)
        ):
            self._check_no_multiple_windows(
                error_case="When using the groupby behavior"
            )

            # The grouping column must be part of the required series
            if group_by_all:
                # group_by_consecutive should be None (checked by asserts above)
                # data should not be a grouped DataFrame (checked by asserts above)
                assert (
                    group_by_all not in self.get_required_series()
                ), "The `group_by_all` column cannot be part of the required series!"
            elif group_by_consecutive:
                # group_by_all should be None (checked by asserts above)
                # data should not be a grouped DataFrame (checked by asserts above)
                assert group_by_consecutive not in self.get_required_series(), (
                    "The `group_by_consecutive` column cannot be part of the required "
                    + "series!"
                )

            # if any of the following params are not None, warn that they won't be of use
            # in the grouped calculation
            ignored_params = [
                ("stride", None),
                ("segment_start_idxs", None),
                ("segment_end_idxs", None),
                ("window_idx", "end"),
                ("include_final_window", False),
            ]
            local_params = locals()

            for ip, default_value in ignored_params:
                if local_params[ip] is not default_value:
                    warnings.warn(
                        f"Parameter `{ip}` will be ignored in case of GroupBy feature"
                        + " calculation."
                    )

            if group_by_consecutive:
                # Strided rollling feature extraction will take place
                return self._calculate_group_by_consecutive(
                    data,
                    group_by_consecutive,
                    return_df,
                    bound_method=bound_method,
                    approve_sparsity=approve_sparsity,
                    show_progress=show_progress,
                    logging_file_path=logging_file_path,
                    n_jobs=n_jobs,
                )
            else:
                # Grouped feature extraction will take place
                if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
                    # group_by_all should not be None (checked by asserts above)
                    # 0. Transform to dataframe
                    series_dict = self._data_to_series_dict(
                        data, self.get_required_series() + [group_by_all]
                    )
                    # 1. Group by `group_by_all` column
                    data = self._group_by_all(series_dict, col_name=group_by_all)

                return self._calculate_group_by_all(
                    data,  # should be a DataFrameGroupBy
                    return_df,
                    show_progress=show_progress,
                    n_jobs=n_jobs,
                    f_handler=f_handler,
                )

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
            self._check_no_multiple_windows(
                error_case="When using both `segment_start_idxs` and `segment_end_idxs`"
            )

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

        return self._calculate_feature_list(
            self._executor_stroll, n_jobs, show_progress, return_df, f_handler
        )

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
            self._check_no_multiple_windows(
                error_case="When reducing a FeatureCollection with manual windows"
            )
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
