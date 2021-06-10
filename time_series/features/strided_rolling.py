"""Contains a (rather) fast implementation of a strided rolling window.

.. todo::
    Add documentation about assumptions/how we index.

"""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import time
from typing import Callable, Union, List, Tuple
from collections import namedtuple

import numpy as np
import pandas as pd

from .function_wrapper import NumpyFuncWrapper
from .logger import logger
from ..utils.data import to_series_list
from ..utils.timedelta import timedelta_to_str


class StridedRolling:
    """Custom time-based sliding window with stride for pandas DataFrames."""
    _NumpySeriesContainer = namedtuple(
        "SeriesContainer", ["values", "start_indexes", "end_indexes"]
    )

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: pd.Timedelta,
        stride: pd.Timedelta,
        window_idx: str = 'end'
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
        window_idx : str
            The window's index position which will be used as index for the
            feature_window aggregation. Must be either of ['begin', 'middle', 'end']

        """
        self.window = window
        self.stride = stride

        # 0. standardize the input
        series_list: List[pd.Series] = to_series_list(data)
        self.series_key: Tuple[str] = tuple([str(s.name) for s in series_list])

        # 1. Determine the tightest bounds
        latest_start = series_list[0].index[0]
        earliest_stop = series_list[0].index[-1]
        for series in series_list[1:]:
            latest_start = max(latest_start, series.index[0])
            earliest_stop = min(earliest_stop, series.index[-1])

        # And slice **all** the series to these tightest bounds
        assert (earliest_stop - latest_start) > window
        series_list = [s[latest_start:earliest_stop] for s in series_list]

        # 2. Create the time_index which will be used for DataFrame reconstruction
        self.index = pd.date_range(latest_start, earliest_stop - window, freq=stride)

        # --- 2. adjust the time_index
        if window_idx == "end":
            self.index += window
        elif window_idx == "middle":
            self.index += window/2
        elif window_idx == "begin":
            pass
        else:
            raise ValueError(f"window index {window_idx} must be either of: "
                             "['end', 'middle', 'begin']")

        # ---------- Efficient numpy code -------
        # 1. Convert everything to int64
        np_latest_start = latest_start.to_datetime64().astype(np.int64)
        np_earliest_stop = earliest_stop.to_datetime64().astype(np.int64)
        np_window = self.window.to_timedelta64().astype(np.int64)
        np_stride = self.stride.to_timedelta64().astype(np.int64)

        start_times = np.arange(
            start=np_latest_start, stop=np_earliest_stop - np_window, step=np_stride
        )

        # TODO should we subtract 1 (for the <= -> < ) in np.searchsorted
        end_times = start_times + np_window

        self.series_containers: List[StridedRolling._NumpySeriesContainer] = []
        for series in series_list:
            np_timestamps = series.index.values.astype(np.int64)
            self.series_containers.append(
                StridedRolling._NumpySeriesContainer(
                    values=series.values,
                    # the slicing will be performed on [ t_start, t_end [
                    start_indexes=np.searchsorted(np_timestamps, start_times, 'left'),
                    # TODO -> maybe hyperparam -> end_boundary -> open/closed
                    #   (default open)
                    end_indexes=np.searchsorted(np_timestamps, end_times, 'left')
                )
            )

    def apply_func(self, np_func: Union[Callable, NumpyFuncWrapper]) -> pd.DataFrame:
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

        Notes
        -----
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
            return f"{'|'.join(self.series_key)}__{feat_name}__{win_stride_str}"

        if not isinstance(np_func, NumpyFuncWrapper):
            np_func = NumpyFuncWrapper(np_func)
        feat_names = np_func.output_names

        t_start = time.time()

        def get_slices(idx):
            # get the slice of each series for the given index
            return [
                sc.values[sc.start_indexes[idx]:sc.end_indexes[idx]]
                for sc in self.series_containers
            ]

        # would be nice if we could optimize this double for loop with something
        # more vectorized
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
            f"{[self.series_key]} with window-stride "
            f"[{self.window}, {self.stride}] in [{elapsed} seconds]!"
        )

        return pd.DataFrame(index=self.index, data=feat_out)
