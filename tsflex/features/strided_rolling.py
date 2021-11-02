"""Contains a (rather) fast implementation of a **time-based** strided rolling window.

.. todo::
    look into **series-based** stroll, instead of np.ndarray based stroll.<br>
    advantages:\n
    * a series is a wrapper around a 1D np.ndarray, so all np-based operations should
      work
    * the end-user can always use the time-index for advanced feature calculation e.g.
      window-based delayed correlation or something like that.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import time
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union, List, Tuple, Optional, Any, TypeVar

import numpy as np
import pandas as pd

from .function_wrapper import FuncWrapper
from .logger import logger
from ..utils.data import SUPPORTED_STROLL_TYPES, to_series_list
from ..utils.time import timedelta_to_str

T = TypeVar('T')  # Declare type variable


class StridedRolling(ABC):
    """Custom time-based sliding window with stride.

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame]
        ``pd.Series`` or ``pd.DataFrame`` to slide over, the index must be a
        (time-zone-aware) ``pd.DatetimeIndex``.
    window : Union[int, pd.Timedelta]
        Either an int or ``pd.Timedelta``, representing the sliding window length in
        samples or the sliding window duration, respectively.
    stride : Union[int, pd.Timedelta]
        Either an int or ``pd.Timedelta``, representing the stride size in samples or
        the stride duration, respectively.
    window_idx : str, optional
        The window's index position which will be used as index for the
        feature_window aggregation. Must be either of: ['begin', 'middle', 'end'], by
        default 'end'.
    bound_method: str, optional
        The start-end bound methodology which is used to generate the slice ranges when
        ``data`` consists of multiple series / columns.
        Must be either of: ['inner', 'outer', 'first'], by default 'inner'.

        * if ``inner``, the inner-bounds of the series are used, the
        * if ``outer``, the inner-bounds of the series are used
        * if ``first``, the first-series it's bound will be used
    approve_sparsity: bool, optional
        Bool indicating whether the user acknowledges that there may be sparsity (i.e.,
        irregularly sampled data), by default False.
        If False and sparsity is observed, a warning is raised.
    data_type: Union[np.array, pd.Series], optional
        The data type of the stroll (either np.array or pd.Series), by default np.array.
        Note: Make sure to only set this argument to pd.Series when this is really 
        required, since pd.Series strided-rolling is significantly less efficient. 
        For a np.array it is possible to create very efficient views, but there is no 
        such thing as a pd.Series view. Thus, for each stroll, a new series is created.

    Notes
    -----
    * This instance withholds a **read-only**-view of the data its values.

    <br>

    .. todo::
        The `bound_method`-argument must still be propagated to `FeatureCollection`

    """
    # Create the named tuple
    _NumpySeriesContainer = namedtuple(
        "SeriesContainer", ["values", "start_indexes", "end_indexes"]
    )

    def __init__(
            self,
            data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
            window: Union[int, float, pd.Timedelta],
            stride: Union[int, float, pd.Timedelta],
            data_type: Optional[Union[np.array, pd.Series]] = np.array,
            window_idx: Optional[str] = "end",
            bound_method: Optional[str] = "inner",
            approve_sparsity: Optional[bool] = False,
    ):
        self.window = window
        self.stride = stride

        self.window_idx = window_idx
        self.approve_sparsity = approve_sparsity
        self.bound_method = bound_method

        # TODO: add support for index-data tuple (faster alternative than pd.Series)
        assert data_type in SUPPORTED_STROLL_TYPES
        self.data_type = data_type

        # 0. Standardize the input
        series_list: List[pd.Series] = to_series_list(data)
        self.series_key: Tuple[str, ...] = tuple([str(s.name) for s in series_list])

        # 1. Determine the bounds
        series_list: List[pd.Series] = to_series_list(data)
        self.t_start, self.t_end = self._determine_bounds(series_list)

        # And slice **all** the series to these tightest bounds
        assert (self.t_end - self.t_start) > window
        if len(series_list) > 1:
            series_list = [s[self.t_start:self.t_end] for s in series_list]

        # 2. Create the new-index which will be used for DataFrame reconstruction
        self.index = self._construct_output_index(series_list[0].index.name)

        # 3. Store the series containers
        np_start_times, np_end_times = self._construct_start_end_times()
        self.series_containers = self._construct_series_containers(
            series_list, np_start_times, np_end_times
        )

        # Finally, check the sparsity assumption
        if not self.approve_sparsity:
            last_container = self.series_containers[-1]
            qs = [0, 0.1, 0.5, 0.9, 1]
            series_idx_stats = np.quantile(
                last_container.end_indexes - last_container.start_indexes, q=qs
            )
            q_str = ", ".join([f"q={q}: {v}" for q, v in zip(qs, series_idx_stats)])
            if not all(series_idx_stats == series_idx_stats[-1]):  # min != max
                warnings.warn(
                    f"There are gaps in the time-series {series_list[-1].name}; "
                    + f"\n \t Quantiles of nb values in window: {q_str}",
                    RuntimeWarning,
                )

    # -------------------------------- PRIVATE METHODS ---------------------------------
    def _determine_bounds(self, series_list: List[pd.Series]) -> Tuple[Any, Any]:
        """Determine the bounds of the passed series.

        Parameters
        ----------
        series_list : List[pd.Series]
            The list of series for which the bounds are determined.

        Returns
        -------
        Tuple[pd.Timestamp, pd.Timestamp]
            The start & end timestamp, respectively.

        """
        if self.bound_method == "inner":
            latest_start = series_list[0].index[0]
            earliest_stop = series_list[0].index[-1]
            for series in series_list[1:]:
                latest_start = max(latest_start, series.index[0])
                earliest_stop = min(earliest_stop, series.index[-1])
            return latest_start, earliest_stop

        if self.bound_method == "outer":
            earliest_start = series_list[0].index[0]
            latest_stop = series_list[0].index[-1]
            for series in series_list[1:]:
                earliest_start = min(earliest_start, series.index[0])
                latest_stop = max(latest_stop, series.index[-1])
            return earliest_start, latest_stop

        elif self.bound_method == "first":
            return series_list[0].index[0], series_list[0].index[-1]

        else:
            raise ValueError(f"invalid bound method string passed {self.bound_method}")

    def _get_window_offset(self, window: T) -> T:
        if self.window_idx == "end":
            return window
        elif self.window_idx == "middle":
            return window / 2
        elif self.window_idx == "begin":
            if isinstance(window, pd.Timedelta):
                return pd.Timedelta(seconds=0)
            return 0
        else:
            raise ValueError(
                f"window index {self.window_idx} must be either of: "
                "['end', 'middle', 'begin']"
            )

    def _construct_series_containers(
            self,
            series_list,
            np_start_times,
            np_end_times
    ) -> List[StridedRolling._NumpySeriesContainer]:

        series_containers: List[StridedRolling._NumpySeriesContainer] = []
        for series in series_list:
            np_idx_times = series.index.values
            if self.data_type is np.array:
                # create a non-writeable view of the series
                series = series.values
                series.flags.writeable = False
            elif self.data_type is pd.Series:
                series.values.flags.writeable = False
                series.index.values.flags.writeable = False
            else:
                raise ValueError("unsupported datatype")

            series_containers.append(
                StridedRolling._NumpySeriesContainer(
                    # TODO: maybe save the pd.Series instead of the np.series
                    values=series,
                    # the slicing will be performed on [ t_start, t_end [
                    # TODO: this can maybe be optimized -> further look into this
                    # np_idx_times, np_start_times, & np_end_times are all sorted!
                    # as we assume & check that the time index is monotonically
                    # increasing & the latter 2 are created using `np.arange()`
                    start_indexes=np.searchsorted(np_idx_times, np_start_times, "left"),
                    end_indexes=np.searchsorted(np_idx_times, np_end_times, "left"),
                )
            )
        return series_containers

    def apply_func(self, func: FuncWrapper) -> pd.DataFrame:
        """Apply a function to the expanded time-series.

        Parameters
        ----------
        func : FuncWrapper
            The Callable wrapped function which will be applied.

        Returns
        -------
        pd.DataFrame
            The merged output of the function applied to every column in a
            new DataFrame. The DataFrame's column-names have the format:
                `<series_col_name(s)>_<feature_name>__w=<window>_s=<stride>`.

        Raises
        ------
        ValueError
            If the passed ``func`` tries to adjust the data its read-only view.

        Notes
        -----
        * If ``func`` is only a callable argument, with no additional logic, this
          will only work for a one-to-one mapping, i.e., no multiple feature-output
          columns are supported for this case!<br>
          If you want to calculate one-to-many, ``func`` should be
          a ``FuncWrapper`` instance and explicitly use
          the ``output_names`` attributes of its constructor.

        """
        feat_names = func.output_names

        t_start = time.time()

        # --- Future work ---
        # would be nice if we could optimize this double for loop with something
        # more vectorized
        out = np.array(
            [func(
                *[sc.values[sc.start_indexes[idx]: sc.end_indexes[idx]]
                  for sc in self.series_containers],
            ) for idx in range(len(self.index))]
        )

        # Aggregate function output in a dictionary
        feat_out = {}
        if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
            assert (
                    len(feat_names) == 1
            ), f"Func {func} returned more than 1 output!"
            feat_out[self._create_feat_col_name(feat_names[0])] = out.flatten()
        if out.ndim == 2 and out.shape[1] > 1:
            assert (
                    len(feat_names) == out.shape[1]
            ), f"Func {func} returned incorrect number of outputs ({out.shape[1]})!"
            for col_idx in range(out.shape[1]):
                feat_out[self._create_feat_col_name(feat_names[col_idx])] = \
                    out[:, col_idx]

        elapsed = time.time() - t_start
        logger.info(
            f"Finished function [{func.func.__name__}] on "
            f"{[self.series_key]} with window-stride "
            f"[{self.window}, {self.stride}] in [{elapsed} seconds]!"
        )

        return pd.DataFrame(index=self.index, data=feat_out)

    # ----------------------------- OVERRIDE THESE METHODS -----------------------------
    @abstractmethod
    def _construct_output_index(self, name: str) -> pd.Index:
        raise NotImplementedError

    @abstractmethod
    def _create_feat_col_name(self, feat_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def _construct_start_end_times(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SequenceStridedRolling(StridedRolling):
    def _construct_output_index(self, name: str) -> pd.Index:
        # 2.1 - and adjust the time_index
        # note: this code can also be placed in the `apply_func` method (if we want to
        #  make the bound window-idx setting feature specific).
        window_idx_offset = self._get_window_offset(self.window)

        # use closed = left to exclude 'end' if it falls on the boundary
        return pd.Index(
            data=np.arange(
                start=self.t_start + window_idx_offset,
                stop=self.t_end - self.window + window_idx_offset,
                step=self.stride,
            ),
            name=name,
        )

    def _create_feat_col_name(self, feat_name: str) -> str:
        win_stride_str = f"w={self.window}_s={self.stride}"
        return f"{'|'.join(self.series_key)}__{feat_name}__{win_stride_str}"

    def _construct_start_end_times(self) -> Tuple[np.ndarray, np.ndarray]:
        # ---------- Efficient numpy code -------
        # 2. Precompute the start & end times (these remain the same for each series)
        # note: this if equivalent to:
        #   if `window` == 'begin":
        #       start_times = self.index.values
        np_start_times = np.arange(
            start=self.t_start, stop=self.t_start + (len(self.index) * self.stride),
            step=self.stride,
            dtype=self.index.dtype,
        )
        np_end_times = np_start_times + self.window
        return np_start_times, np_end_times


class TimeStridedRolling(StridedRolling):
    def _construct_output_index(self, name: str) -> pd.DatetimeIndex:
        # 2.1 - and adjust the time_index
        # note: this code can also be placed in the `apply_func` method (if we want to
        #  make the bound window-idx setting feature specific).
        window_idx_offset = self._get_window_offset(self.window)

        # use closed = left to exclude 'end' if it falls on the boundary
        # note: the index automatically takes the timezone of `t_start` & `t_end`
        # note: the index-name of the first passed series will be used
        return pd.date_range(
            start=self.t_start + window_idx_offset,
            end=self.t_end - self.window + window_idx_offset,
            freq=self.stride,
            name=name
        )

    def _create_feat_col_name(self, feat_name: str) -> str:
        # Convert win & stride to time-string if available :)
        win_str = timedelta_to_str(self.window)
        stride_str = timedelta_to_str(self.stride)
        win_stride_str = f"w={win_str}_s={stride_str}"
        return f"{'|'.join(self.series_key)}__{feat_name}__{win_stride_str}"

    def _construct_start_end_times(self) -> Tuple[np.ndarray, np.ndarray]:
        # ---------- Efficient numpy code -------
        # 1. Convert everything to int64
        np_start = self.t_start.to_datetime64()
        np_window = self.window.to_timedelta64()
        np_stride = self.stride.to_timedelta64()

        # 2. Precompute the start & end times (these remain the same for each series)
        # note: this if equivalent to:
        #   if `window` == 'begin":
        #       start_times = self.index.values
        np_start_times = np.arange(
            start=np_start, stop=np_start + (len(self.index) * np_stride),
            step=np_stride,
            dtype=np.datetime64,
        )
        np_end_times = np_start_times + np_window
        return np_start_times, np_end_times
