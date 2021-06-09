"""Contains a (rather) fast implementation of a strided rolling window."""

__author__ = "Vic Degraeve, Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import time
from typing import Callable, Union, List

import numpy as np
import pandas as pd

from .function_wrapper import NumpyFuncWrapper
from .logger import logger
from ..utils.data import to_series_list
from ..utils.timedelta import timedelta_to_str


class StridedRolling:
    """Custom time-based sliding window with stride for pandas DataFrames."""

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: pd.Timedelta,
        stride: pd.Timedelta
    ):
        """Create a time-based StridedRolling object.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame]
            :class:`pd.Series` or :class:`pd.DataFrame` to slide over, the index must
            be a (time-zone-aware) `pd.DatetimeIndex`.
        window : Union[int, pd.Timedelta]
            Either an int or `pd.Timedelta`, representing the sliding window length in
            samples or the sliding window duration, respectively.
        stride : Union[int, pd.Timedelta]
            Either an int or `pd.Timedelta`, representing the stride size in samples or
            the stride duration, respectively.

        Note
        -----
        Time based window-stride parameters will be converted into integers at inference
        time. The integer conversion will use flooring, and can be written as the
        following formula:

            time_parameter // series_period

        This also implies that **we must be able to infer the freq** from `data` when
        time-based window-stride parameters are passed.

        """
        # Store the orig input (can be pd.Timedelta)
        self.window = window
        self.stride = stride

        series_list = to_series_list(data)
        if len(series_list) == 1:
            self._key = tuple([str(series_list[0].name)])
        elif isinstance(series_list, list):
            self._key = tuple([str(s.name) for s in series_list])

        # Determine the tightest bounds
        self._latest_start = series_list[0].index[0]
        self._earliest_stop = series_list[0].index[-1]
        for series in series_list[1:]:
            self._latest_start = max(self._latest_start, series.index[0])
            self._earliest_stop = min(self._earliest_stop, series.index[-1])

        # Slice the series to the tightest bounds
        assert (self._earliest_stop - self._latest_start) > window
        series_list = [s[self._latest_start:self._earliest_stop] for s in series_list]

        # create the time_index
        # TODO check
        self.index = pd.date_range(
            self._latest_start, self._earliest_stop - window, freq=stride)

        # adjust the time_index
        # TODO check
        window_idx = 'last'
        if window_idx == "last":
            self.index += window
        elif window_idx == "middle":
            self.index += window/2
        elif window_idx == "first":
            pass

        self._values, self._start_idxs, self._end_idxs = [], [], []
        for series in series_list:
            start_idxs, end_idxs = self._get_slice_indexes(series.index)
            self._start_idxs.append(start_idxs)
            self._end_idxs.append(end_idxs)
            self._values.append(series.values)

    def _get_slice_indexes(self, series_index: pd.DatetimeIndex):
        t_start = self._latest_start.to_datetime64().astype(np.int64)
        t_end = self._earliest_stop.to_datetime64().astype(np.int64)

        np_timestamps = series_index.values.astype(np.int64)
        np_window = self.window.to_timedelta64().astype(np.int64)
        np_stride = self.stride.to_timedelta64().astype(np.int64)

        # todo -> blijft hetzelfde voor alle series
        start_times = np.arange(t_start, t_end - np_window, step=np_stride)

        start_idxs = np.searchsorted(np_timestamps, start_times, 'left')
        end_idxs = np.searchsorted(np_timestamps, start_times + np_window, 'right')
        return start_idxs, end_idxs

    def apply_func(self, np_func: Union[Callable, NumpyFuncWrapper], single_series_func=False) -> pd.DataFrame:
        """Apply a function to the expanded time-series.

        Parameters
        ----------
        np_func : Union[Callable, NumpyFuncWrapper]
            The Callable (wrapped) function which will be applied.

        Returns
        -------
        pd.DataFrame
            The merged output of the function applied to every column in a
            new DataFrame. The DataFrame's column-names have the format:
                `<series_col_name(s)>_<feature_name>__w=<window>_s=<stride>`.

        Note
        ----
        * If `np_func` is only a callable argument, with no additional logic, this
            will only work for a one-to-one mapping, i.e., no multiple feature-output
            columns are supported for this case!
        * If you want to calculate one-to-many -> `np_func` should be
             a `NumpyFuncWrapper` instance and explicitly use
             the `output_names` attributes of its constructor.

        """
        # Convert win & stride to time-string if available :)
        def create_feat_col_name(feat_name) -> str:
            win_str = timedelta_to_str(self.window)
            stride_str = timedelta_to_str(self.stride)
            win_stride_str = f"w={win_str}_s={stride_str}"
            return f"{'|'.join(self._key)}__{feat_name}__{win_stride_str}"

        if not isinstance(np_func, NumpyFuncWrapper):
            np_func = NumpyFuncWrapper(np_func)
        feat_names = np_func.output_names

        t_start = time.time()

        def get_slices(idx):
            # get the slice of each series for the given index
            return [
                self._values[i][self._start_idxs[i][idx]:self._end_idxs[i][idx]]
                for i in range(len(self._key))
            ]

        out = np.array([np_func(*get_slices(idx)) for idx in range(len(self.index))])

        # Aggregate function output in a dictionary
        feat_out = {}
        if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
            assert len(feat_names) == 1
            feat_out[create_feat_col_name(feat_names[0])] = out.flatten()
        if out.ndim == 2 and out.shape[1] > 1:
            assert len(feat_names) == out.shape[1]
            for col_idx in range(out.shape[1]):
                feat_out[create_feat_col_name(feat_names[col_idx])] = out[:, col_idx]

        elapsed = time.time() - t_start
        logger.info(
            f"Finished function [{np_func.func.__name__}] on "
            f"{[self._key]} with window-stride "
            f"[{self.window}, {self.stride}] in [{elapsed} seconds]!"
        )

        return pd.DataFrame(index=self.index, data=feat_out)
