"""
Withholds a (rather) fast implementation of an **index-based** strided rolling window.

.. TODO::

    Look into the implementation of a new func-input-data-type that is a
    Tuple[index, values]. This should be multitudes faster than using the
    series-datatype and the user can still leverage the index-awareness of the values.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt"

import time
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union, List, Tuple, Optional, TypeVar

import numpy as np
import pandas as pd

from ..function_wrapper import FuncWrapper
from ..logger import logger
from ..utils import _determine_bounds
from ...utils.data import SUPPORTED_STROLL_TYPES, to_series_list
from ...utils.attribute_parsing import DataType, AttributeParser
from ...utils.time import timedelta_to_str

# Declare a type variable
T = TypeVar("T")


class StridedRolling(ABC):
    """Custom time-based sliding window with stride.

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame]
        ``pd.Series`` or ``pd.DataFrame`` to slide over, the index must be either
        numeric or a ``pd.DatetimeIndex``.
    window : Union[float, pd.Timedelta]
        Either an int, float, or ``pd.Timedelta``, representing the sliding window size
        in terms of the index (in case of a int or float) or the sliding window duration
        (in case of ``pd.Timedelta``).
    stride : Union[float, pd.Timedelta]
        Either an int, float, or ``pd.Timedelta``, representing the stride size in terms
        of the index (in case of a int or float) or the stride duration (in case of
        ``pd.Timedelta``).
    start_idx: Union[float, pd.Timestamp], optional
        The start-index which will be used for each series passed to `data`. This is
        especially useful if multiple `StridedRolling` instances are created and the
        user want to ensure same (start-)indexes for each of them.
    end_idx: Union[float, pd.Timestamp], optional
        The end-index which will be used as sliding end-limit for each series passed to
        `data`.
    func_data_type: Union[np.array, pd.Series], optional
        The data type of the stroll (either np.array or pd.Series), by default np.array.
        <br>
        .. Note::
            Make sure to only set this argument to pd.Series when this is really
            required, since pd.Series strided-rolling is significantly less efficient.
            For a np.array it is possible to create very efficient views, but there is no
            such thing as a pd.Series view. Thus, for each stroll, a new series is created.
    window_idx : str, optional
        The window's index position which will be used as index for the
        feature_window aggregation. Must be either of: `["begin", "middle", "end"]`, by
        default "end".
    approve_sparsity: bool, optional
        Bool indicating whether the user acknowledges that there may be sparsity (i.e.,
        irregularly sampled data), by default False.
        If False and sparsity is observed, a warning is raised.

    Notes
    -----
    * This instance withholds a **read-only**-view of the data its values.

    """

    # Class variables which are used by sub-classes
    win_str_type: DataType
    reset_series_index_b4_segmenting: bool = False

    # Create the named tuple
    _NumpySeriesContainer = namedtuple(
        "SeriesContainer", ["name", "values", "start_indexes", "end_indexes"]
    )

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: T,
        stride: T,
        start_idx: Optional[T] = None,
        end_idx: Optional[T] = None,
        func_data_type: Optional[Union[np.array, pd.Series]] = np.array,
        window_idx: Optional[str] = "end",
        approve_sparsity: Optional[bool] = False,
    ):
        self.window = window
        self.stride = stride

        # Note: the sub-class should set the self.func_data_type attribute
        assert AttributeParser.check_expected_type([window, stride], self.win_str_type)

        self.window_idx = window_idx
        self.approve_sparsity = approve_sparsity

        assert func_data_type in SUPPORTED_STROLL_TYPES
        self.data_type = func_data_type

        # 0. Standardize the input
        series_list: List[pd.Series] = to_series_list(data)
        self.series_dtype = AttributeParser.determine_type(series_list)
        self.series_key: Tuple[str, ...] = tuple([str(s.name) for s in series_list])

        # 1. Determine the start index
        self.start, self.end = start_idx, end_idx
        if self.start is None or self.end is None:
            start, end = _determine_bounds("inner", series_list)

            # update self.start & self.end if it was not passed
            self.start = start if self.start is None else self.start
            self.end = end if self.end is None else self.end

        # Especially useful when the index dtype differs from the win-stride-dtype
        # e.g. -> performing a int-based stroll on time-indexed data
        # Note: this is very niche and thus requires advanced knowledge
        self._update_start_end_indices_to_stroll_type(series_list)

        # 2. Create a new-index which will be used for DataFrame reconstruction
        # Note: the index-name of the first passed series will be re-used as index-name
        self.index = self._construct_output_index(series_list[0])

        # 3. Construct the index ranges and store the series containers
        np_start_times, np_end_times = self._construct_start_end_times()
        self.series_containers = self._construct_series_containers(
            series_list, np_start_times, np_end_times
        )

        # 4. Check the sparsity assumption
        if not self.approve_sparsity:
            qs = [0, 0.1, 0.5, 0.9, 1]
            for container in self.series_containers:
                series_idx_stats = np.quantile(
                    container.end_indexes - container.start_indexes, q=qs
                )
                q_str = ", ".join([f"q={q}: {v}" for q, v in zip(qs, series_idx_stats)])
                # Warn when min != max
                if not all(series_idx_stats == series_idx_stats[-1]):
                    warnings.warn(
                        f"There are gaps in the sequence of the {container.name}"
                        f"-series;\n \t Quantiles of nb values in window: {q_str}",
                        RuntimeWarning,
                    )

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
        self, series_list, np_start_times, np_end_times
    ) -> List[StridedRolling._NumpySeriesContainer]:

        series_containers: List[StridedRolling._NumpySeriesContainer] = []
        for series in series_list:
            if not self.reset_series_index_b4_segmenting:
                np_idx_times = series.index.values
            else:
                np_idx_times = np.arange(len(series))

            series_name = series.name
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
                    name=series_name,
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
        """Apply a function to the segmented series.

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
        * If you want to calculate one-to-many, ``func`` should be
          a ``FuncWrapper`` instance and explicitly use
          the ``output_names`` attributes of its constructor.

        """
        feat_names = func.output_names

        t_start = time.time()

        # --- Future work ---
        # would be nice if we could optimize this double for loop with something
        # more vectorized
        #
        # As for now we use a map to apply the function (as this evaluates its
        # expression only once, whereas a list comprehension evaluates its expression
        # every time).
        # See more why: https://stackoverflow.com/a/59838723
        out = np.array(
            list(
                map(
                    func,
                    *[
                        [
                            sc.values[sc.start_indexes[idx]: sc.end_indexes[idx]]
                            for idx in range(len(self.index))
                        ]
                        for sc in self.series_containers
                    ],
                )
            )
        )

        # Aggregate function output in a dictionary
        feat_out = {}
        if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
            assert len(feat_names) == 1, f"Func {func} returned more than 1 output!"
            feat_out[self._create_feat_col_name(feat_names[0])] = out.flatten()
        if out.ndim == 2 and out.shape[1] > 1:
            assert (
                len(feat_names) == out.shape[1]
            ), f"Func {func} returned incorrect number of outputs ({out.shape[1]})!"
            for col_idx in range(out.shape[1]):
                feat_out[self._create_feat_col_name(feat_names[col_idx])] = out[
                    :, col_idx
                ]

        elapsed = time.time() - t_start
        logger.info(
            f"Finished function [{func.func.__name__}] on "
            f"{[self.series_key]} with window-stride "
            f"[{self.window}, {self.stride}] in [{elapsed} seconds]!"
        )

        return pd.DataFrame(index=self.index, data=feat_out)

    def _update_start_end_indices_to_stroll_type(self, series_list: List[pd.Series]):
        pass

    # ----------------------------- OVERRIDE THESE METHODS -----------------------------
    @abstractmethod
    def _construct_output_index(self, series: pd.Series) -> pd.Index:
        raise NotImplementedError

    @abstractmethod
    def _construct_start_end_times(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _create_feat_col_name(self, feat_name: str) -> str:
        raise NotImplementedError


class SequenceStridedRolling(StridedRolling):
    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: float,
        stride: float,
        *args,
        **kwargs,
    ):
        self.win_str_type = DataType.SEQUENCE
        super().__init__(data, window, stride, *args, **kwargs)

    # ------------------------------ Overridden methods ------------------------------
    def _construct_output_index(self, series: pd.Series) -> pd.Index:
        window_offset = self._get_window_offset(self.window)
        # bool which indicates whether the `end` lies on the boundary
        # and as arange does not include the right boundary -> use it to enlarge `stop`
        boundary = (self.end - self.start - self.window) % self.stride == 0
        return pd.Index(
            data=np.arange(
                start=self.start + window_offset,
                stop=self.end - self.window + window_offset + self.stride * boundary,
                step=self.stride,
            ),
            name=series.index.name,
        )

    def _construct_start_end_times(self) -> Tuple[np.ndarray, np.ndarray]:
        # ---------- Efficient numpy code -------
        # 1. Precompute the start & end times (these remain the same for each series)
        # note: this if equivalent to:
        #   if `window` == 'begin":
        #       start_times = self.index.values
        np_start_times = np.arange(
            start=self.start,
            stop=self.start + (len(self.index) * self.stride),
            step=self.stride,
            dtype=self.index.dtype,
        )
        np_end_times = np_start_times + self.window
        return np_start_times, np_end_times

    def _create_feat_col_name(self, feat_name: str) -> str:
        # TODO -> this is not that loosely coupled if we want somewhere else in the code
        # to also reproduce col-name construction
        win_stride_str = f"w={self.window}_s={self.stride}"
        return f"{'|'.join(self.series_key)}__{feat_name}__{win_stride_str}"


class TimeStridedRolling(StridedRolling):
    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: pd.Timedelta,
        stride: pd.Timedelta,
        *args,
        **kwargs,
    ):
        self.win_str_type = DataType.TIME
        super().__init__(data, window, stride, *args, **kwargs)

    # ---------------------------- Overridden methods ------------------------------
    def _construct_output_index(self, series: pd.Series) -> pd.DatetimeIndex:
        # note: the index automatically takes the timezone of `start` & `end`
        window_offset = self._get_window_offset(self.window)
        return pd.date_range(
            start=self.start + window_offset,
            end=self.end - self.window + window_offset,
            freq=self.stride,
            name=series.index.name,
            closed=None,
        )

    def _construct_start_end_times(self) -> Tuple[np.ndarray, np.ndarray]:
        # ---------- Efficient numpy code -------
        # 1. Convert everything to int64
        np_start = self.start.to_datetime64()
        np_window = self.window.to_timedelta64()
        np_stride = self.stride.to_timedelta64()

        # 2. Precompute the start & end times (these remain the same for each series)
        # note: this if equivalent to:
        #   if `window` == 'begin":
        #       start_times = self.index.values
        np_start_times = np.arange(
            start=np_start,
            stop=np_start + (len(self.index) * np_stride),
            step=np_stride,
            dtype=np.datetime64,
        )
        np_end_times = np_start_times + np_window
        return np_start_times, np_end_times

    def _create_feat_col_name(self, feat_name: str) -> str:
        # Convert win & stride to time-string if available :)
        win_str = timedelta_to_str(self.window)
        stride_str = timedelta_to_str(self.stride)
        win_stride_str = f"w={win_str}_s={stride_str}"
        return f"{'|'.join(self.series_key)}__{feat_name}__{win_stride_str}"


class TimeIndexSampleStridedRolling(SequenceStridedRolling):
    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: int,
        stride: int,
        *args,
        **kwargs,
    ):
        """
        .. Warning::
            When `data` consists of multiple independently sampled series
            (e.g. feature functions which take multiple series as input),
            The time-**index of each series**: \n
            - must _roughly_ **share** the same **sample frequency**.
            - will be first time-aligned before transitioning to sample-segmentation by
              using the inner bounds

        """
        # We want to reset the index as its type differs from the passed win-stride
        # configs
        self.reset_series_index_b4_segmenting = True

        series_list = to_series_list(data)
        if isinstance(data, list) and len(data) > 1:
            # Slice data into its inner range so that the start position
            # is aligned (when we will use sample-based methodologies)
            start, end = _determine_bounds("inner", series_list)
            series_list = [s[start:end] for s in series_list]
            kwargs.update({"start_idx": start, "end_idx": end})

        # pass the sliced series list instead of data
        super().__init__(series_list, window, stride, *args, **kwargs)

        assert self.series_dtype == DataType.TIME

        # we want to assure that the window-stride arguments are integers (samples)
        assert all(isinstance(p, int) for p in [self.window, self.stride])

    # ---------------------------- Overridden methods ------------------------------
    def _update_start_end_indices_to_stroll_type(self, series_list: List[pd.Series]):
        # update the start and end times to the sequence datatype
        self.start, self.end = np.searchsorted(
            series_list[0].index.values,
            [self.start.to_datetime64(), self.end.to_datetime64()],
            "left",
        )

    def _construct_output_index(self, series: pd.Series) -> pd.DatetimeIndex:
        window_offset = int(self._get_window_offset(self.window))
        assert all(isinstance(p, np.int64) for p in [self.start, self.end])

        # Note: so we have one or multiple time-indexed series on which we specified a
        # sample based window-stride configuration -> assumptions we make
        # * if we have  multiple series as input for a feature-functions
        #  -> we assume the time-indexes are (roughly) the same for each series

        # bool which indicates whether the `end` lies on the boundary
        # and as arange does not include the right boundary -> use it to enlarge `stop`
        boundary = (self.end - self.start - self.window) % self.stride == 0

        return series.iloc[
            np.arange(
                start=int(window_offset),
                stop=self.end - self.window + window_offset + self.stride * boundary,
                step=self.stride,
            )
        ].index

    def _construct_start_end_times(self) -> Tuple[np.ndarray, np.ndarray]:
        # ---------- Efficient numpy code -------
        # 1. Precompute the start & end times (these remain the same for each series)
        # note: this if equivalent to:
        #   if `window` == 'begin":
        #       start_times = self.index.values
        np_start_times = np.arange(
            start=self.start,
            stop=self.start + (len(self.index) * self.stride),
            step=self.stride,
            dtype=np.int64,  # the only thing that is changed w.r.t. the Superclass
        )
        np_end_times = np_start_times + self.window
        return np_start_times, np_end_times
