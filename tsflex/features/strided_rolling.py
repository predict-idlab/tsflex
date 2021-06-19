"""Contains a (rather) fast implementation of a **time-based** strided rolling window.

"""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import time
import warnings
import numpy as np
import pandas as pd

from typing import Callable, Union, List, Tuple, Optional
from collections import namedtuple

from .function_wrapper import NumpyFuncWrapper
from .logger import logger
from ..utils.data import to_series_list
from ..utils.timedelta import timedelta_to_str


class StridedRolling:
    """Custom time-based sliding window with stride for pandas DataFrames.

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

    Notes
    -----
    * This instance withholds a **read-only**-view of the data its values.

    <br>

    .. todo::
        The `bound_method` must still be propagated to the `FeatureCollection`-class.

    """

    # Create the named tuple
    _NumpySeriesContainer = namedtuple(
        "SeriesContainer", ["values", "start_indexes", "end_indexes"]
    )

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: pd.Timedelta,
        stride: pd.Timedelta,
        window_idx: Optional[str] = "end",
        bound_method: Optional[str] = "inner",
        approve_sparsity: Optional[bool] = False,
    ):
        self.window: pd.Timedelta = window
        self.stride: pd.Timedelta = stride

        # 0. standardize the input
        series_list: List[pd.Series] = to_series_list(data)
        self.series_key: Tuple[str] = tuple([str(s.name) for s in series_list])

        # 1. Determine the bounds
        t_start, t_end = self._determine_bounds(series_list, bound_method)

        # And slice **all** the series to these tightest bounds
        assert (t_end - t_start) > window
        series_list = [s[t_start:t_end] for s in series_list]

        # 2. Create the time_index which will be used for DataFrame reconstruction
        # use closed = left to exclude 'end' if it falls on the boundary
        # note: the index automatically takes the timezone of `t_start` & `t_end`
        # note: the index-name of the first passed series will be used
        self.index = pd.date_range(
            t_start, t_end - window, freq=stride, name=series_list[0].index.name
        )

        # --- and adjust the time_index
        # note: this code can also be placed in the `apply_func` method (if we want to
        #  make the bound window-idx setting feature specific).
        if window_idx == "end":
            self.index += window
        elif window_idx == "middle":
            self.index += window / 2
        elif window_idx == "begin":
            pass
        else:
            raise ValueError(
                f"window index {window_idx} must be either of: "
                "['end', 'middle', 'begin']"
            )

        # ---------- Efficient numpy code -------
        # 1. Convert everything to int64
        np_start = t_start.to_datetime64().astype(np.int64)
        np_window = self.window.to_timedelta64().astype(np.int64)
        np_stride = self.stride.to_timedelta64().astype(np.int64)

        # 2. Precompute the start & end times (these remain the same for each series)
        # note: this if equivalent to:
        #   if `window` == 'begin":
        #       start_times = self.index.values.astype(np.int64)
        np_start_times = np.arange(
            start=np_start, stop=np_start + len(self.index)*np_stride, step=np_stride,
            dtype=np.int64,
        )
        np_end_times = np_start_times + np_window

        self.series_containers: List[StridedRolling._NumpySeriesContainer] = []
        for series in series_list:
            # create a non-writeable view of the series
            np_series = series.values
            np_series.flags.writeable = False

            np_idx_times = series.index.values.astype(np.int64)
            self.series_containers.append(
                StridedRolling._NumpySeriesContainer(
                    values=np_series,
                    # the slicing will be performed on [ t_start, t_end [
                    # TODO: this can mabye be optimized -> further look into this
                    # np_idx_times, np_start_times, & np_end_times are all sorted!
                    # as we assume & check that the time index is monotonically
                    # increasing & the latter 2 are created using `np.arange()`
                    start_indexes=np.searchsorted(np_idx_times, np_start_times, "left"),
                    end_indexes=np.searchsorted(np_idx_times, np_end_times, "left"),
                )
            )

            if not approve_sparsity:
                last_container = self.series_containers[-1]
                qs = [0, 0.1, 0.5, 0.9, 1]
                series_idx_stats = np.quantile(
                    last_container.end_indexes - last_container.start_indexes, q=qs
                )
                q_str = ", ".join([f"q={q}: {v}" for q, v in zip(qs, series_idx_stats)])
                if series_idx_stats[0] != series_idx_stats[1]:  # min != max
                    warnings.warn(
                        f"There are gaps in the time-series {series.name}; "
                        + f"\n \t Quantiles of nb values in window: {q_str}",
                        RuntimeWarning,
                    )

    @staticmethod
    def _determine_bounds(
        series_list: List[pd.Series], bound_method: str
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Determine the bounds of the passed series.

        Parameters
        ----------
        series_list : List[pd.Series]
            The list of series for which the bounds are determined.

        bound_method : str
            The methodology which is used for the ``series_list`` bound determination

        Returns
        -------
        Tuple[pd.Timestamp, pd.Timestamp]
            The start & end timestamp, respectively.

        """
        if bound_method == "inner":
            latest_start = series_list[0].index[0]
            earliest_stop = series_list[0].index[-1]
            for series in series_list[1:]:
                latest_start = max(latest_start, series.index[0])
                earliest_stop = min(earliest_stop, series.index[-1])
            return latest_start, earliest_stop

        if bound_method == "outer":
            earliest_start = series_list[0].index[0]
            latest_stop = series_list[0].index[-1]
            for series in series_list[1:]:
                earliest_start = min(earliest_start, series.index[0])
                latest_stop = max(latest_stop, series.index[-1])
            return earliest_start, latest_stop

        elif bound_method == "first":
            return series_list[0].index[0], series_list[0].index[-1]

        else:
            raise ValueError(f"invalid bound method string passed {bound_method}")

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

        Raises
        ------
        ValueError
            If the passed ``np_func`` tries to adjust the data its read-only view.


        Notes
        -----
        * If ``np_func`` is only a callable argument, with no additional logic, this
          will only work for a one-to-one mapping, i.e., no multiple feature-output
          columns are supported for this case!<br>
          If you want to calculate one-to-many, ``np_func`` should be
          a ``NumpyFuncWrapper`` instance and explicitly use
          the ``output_names`` attributes of its constructor.

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
                sc.values[sc.start_indexes[idx] : sc.end_indexes[idx]]
                for sc in self.series_containers
            ]

        # --- Future work ---
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
