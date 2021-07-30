"""Contains a (rather) fast implementation of a **time-based** strided rolling window.

.. todo::
    look into **series-based** stroll, instead of np.ndarray based stroll.<br>
    advantages:\n
    * a series is a wrapper around a 1D np.ndarray, so all np-based operations should
      work
    * the end-user can always use the time-index for advanced feature calculation e.g.
      window-based delayed correlation or something like that.

"""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import time
import warnings
import pandas as pd
import numpy as np

from collections import namedtuple
from typing import Callable, Union, List, Tuple, Optional

from .function_wrapper import FuncWrapper
from .logger import logger
from ..utils.data import SUPPORTED_STROLL_TYPES, to_series_list
from ..utils.time import timedelta_to_str


class StridedRolling:
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
            window: pd.Timedelta,
            stride: pd.Timedelta,
            data_type: Optional[Union[np.array, pd.Series]] = np.array,
            window_idx: Optional[str] = "end",
            bound_method: Optional[str] = "inner",
            approve_sparsity: Optional[bool] = False,
    ):
        self.window: pd.Timedelta = window
        self.stride: pd.Timedelta = stride
        
        assert data_type in SUPPORTED_STROLL_TYPES
        self.data_type = data_type

        # 0. standardize the input
        series_list: List[pd.Series] = to_series_list(data)
        self.series_key: Tuple[str, ...] = tuple([str(s.name) for s in series_list])

        # 1. Determine the bounds
        t_start, t_end = self._determine_bounds(series_list, bound_method)

        # And slice **all** the series to these tightest bounds
        assert (t_end - t_start) > window
        if len(series_list) > 1:
            series_list = [s[t_start:t_end] for s in series_list]

        # 2. Create the time_index which will be used for DataFrame reconstruction
        # 2.1 - and adjust the time_index
        # note: this code can also be placed in the `apply_func` method (if we want to
        #  make the bound window-idx setting feature specific).
        if window_idx == "end":
            window_idx_offset = window
        elif window_idx == "middle":
            window_idx_offset = window / 2
        elif window_idx == "begin":
            window_idx_offset = pd.Timedelta(seconds=0)
        else:
            raise ValueError(
                f"window index {window_idx} must be either of: "
                "['end', 'middle', 'begin']"
            )

        # use closed = left to exclude 'end' if it falls on the boundary
        # note: the index automatically takes the timezone of `t_start` & `t_end`
        # note: the index-name of the first passed series will be used
        self.index = pd.date_range(
            start=t_start + window_idx_offset,
            end=t_end - window + window_idx_offset,
            freq=stride,
            name=series_list[0].index.name
        )

        # ---------- Efficient numpy code -------
        # 1. Convert everything to int64
        np_start = t_start.to_datetime64()
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

        self.series_containers: List[StridedRolling._NumpySeriesContainer] = []
        for series in series_list:
            np_idx_times = series.index.values
            series_name = series.name
            if data_type is np.array:
                # create a non-writeable view of the series
                series = series.values
                series.flags.writeable = False
            elif data_type is pd.Series:
                series.values.flags.writeable = False
                series.index.values.flags.writeable = False
            else:
                raise ValueError("unsupported datatype")

            self.series_containers.append(
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

            if not approve_sparsity:
                last_container = self.series_containers[-1]
                qs = [0, 0.1, 0.5, 0.9, 1]
                series_idx_stats = np.quantile(
                    last_container.end_indexes - last_container.start_indexes, q=qs
                )
                q_str = ", ".join([f"q={q}: {v}" for q, v in zip(qs, series_idx_stats)])
                if not all(series_idx_stats == series_idx_stats[-1]):  # min != max
                    warnings.warn(
                        f"There are gaps in the time-series {series_name}; "
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

        # Convert win & stride to time-string if available :)
        def create_feat_col_name(feat_name) -> str:
            win_str = timedelta_to_str(self.window)
            stride_str = timedelta_to_str(self.stride)
            win_stride_str = f"w={win_str}_s={stride_str}"
            return f"{'|'.join(self.series_key)}__{feat_name}__{win_stride_str}"

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
            feat_out[create_feat_col_name(feat_names[0])] = out.flatten()
        if out.ndim == 2 and out.shape[1] > 1:
            assert (
                len(feat_names) == out.shape[1]
            ), f"Func {func} returned incorrect number of outputs ({out.shape[1]})!"
            for col_idx in range(out.shape[1]):
                feat_out[create_feat_col_name(feat_names[col_idx])] = out[:, col_idx]

        elapsed = time.time() - t_start
        logger.info(
            f"Finished function [{func.func.__name__}] on "
            f"{[self.series_key]} with window-stride "
            f"[{self.window}, {self.stride}] in [{elapsed} seconds]!"
        )

        return pd.DataFrame(index=self.index, data=feat_out)
