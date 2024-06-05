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
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from ...utils.argument_parsing import timedelta_to_str
from ...utils.attribute_parsing import AttributeParser, DataType
from ...utils.data import SUPPORTED_STROLL_TYPES, to_list, to_series_list, to_tuple
from ..function_wrapper import FuncWrapper
from ..utils import (
    _check_start_end_array,
    _determine_bounds,
    _log_func_execution,
    _process_func_output,
)

# Declare a type variable
T = TypeVar("T", int, float, pd.Timedelta)


class StridedRolling(ABC):
    """Custom time-based sliding window with stride.

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
        ``pd.Series`` or ``pd.DataFrame`` to slide over, the index must be either
        numeric or a ``pd.DatetimeIndex``.
    window : Union[float, pd.Timedelta]
        Either an int, float, or ``pd.Timedelta``, representing the sliding window size
        in terms of the index (in case of a int or float) or the sliding window duration
        (in case of ``pd.Timedelta``).
    strides : Union[float, pd.Timedelta, List[Union[float, pd.Timedelta]]], optional
        Either a list of int, float, or ``pd.Timedelta``, representing the stride sizes
        in terms of the index (in case of a int or float) or the stride duration (in
        case of ``pd.Timedelta``). By default None.
    segment_start_idxs: np.ndarray, optional
        The start indices for the segmented windows. If not provided, the start indices
        will be computed from the data using the passed ``strides`` or by using the
        ``segment_end_idxs`` (if not none) + ``window``. By default None.
    segment_end_idxs: np.ndarray, optional
        The end indices for the segmented windows. If not provided, the end indices will
        be computed from either (1) the data using the passed ``window`` + ``strides``
        or (2) the ``segment_start_idxs`` + ``window``, By default None.
        .. Note::
            When you pass arrays to both ``segment_start_idxs`` and
            ``segment_end_idxs``, the corresponding index-values of these arrays will be
            used as segment-ranges. As a result, the following properties must be met:\n
              - both arrays should have equal length
              - all values in ``segment_start_idxs`` should be <= ``segment_end_idxs``
    start_idx: Union[float, pd.Timestamp], optional
        The start-index which will be used for each series passed to `data`. This is
        especially useful if multiple ``StridedRolling`` instances are created and the
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
            For a np.array it is possible to create very efficient views, but there is
            no such thing as a pd.Series view. Thus, for each stroll, a new series is
            created, inducing a lot of non-feature calculation of overhead.
    window_idx : str, optional
        The window's index position which will be used as index for the
        feature_window aggregation. Must be either of: `["begin", "middle", "end"]`, by
        default "end".
    include_final_window: bool, optional
        Whether the final (possibly incomplete) window should be included in the
        strided-window segmentation, by default False.

        .. Note::
            The remarks below apply when ``include_final_window`` is set to True.
            The user should be aware that the last window *might* be incomplete, i.e.;

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
            - *(len * sampling_rate - window_size) % stride = 0*. Remark that the above
              case is a base case of this.
    approve_sparsity: bool, optional
        Bool indicating whether the user acknowledges that there may be sparsity (i.e.,
        irregularly sampled data), by default False.
        If False and sparsity is observed, a warning is raised.

    Notes
    -----
    * This instance withholds a **read-only**-view of the data its values.

    """

    # Class variables which are used by subclasses
    win_str_type: DataType
    reset_series_index_b4_segmenting: bool = False
    OUTSIDE_DATA_BOUNDS_WARNING: str = (
        "Some segment indexes are outside the range of the data its index."
    )

    # Create the named tuple
    _NumpySeriesContainer = namedtuple(  # type: ignore[name-match]
        "SeriesContainer", ["name", "values", "start_indexes", "end_indexes"]
    )

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: Optional[T],
        strides: Optional[Union[T, List[T]]] = None,
        segment_start_idxs: Optional[np.ndarray] = None,
        segment_end_idxs: Optional[np.ndarray] = None,
        start_idx: Optional[T] = None,
        end_idx: Optional[T] = None,
        func_data_type: Union[np.ndarray, pd.Series] = np.ndarray,
        window_idx: str = "end",
        include_final_window: bool = False,
        approve_sparsity: bool = False,
    ):
        if strides is not None:
            strides = to_list(strides)

        # Check the passed segment indices
        if segment_start_idxs is not None and segment_end_idxs is not None:
            _check_start_end_array(segment_start_idxs, segment_end_idxs)

        if window is not None:
            assert AttributeParser.check_expected_type(
                [window] + ([] if strides is None else strides), self.win_str_type
            )

        self.window = window  # type: ignore[var-annotated]
        self.strides = strides  # type: ignore[var-annotated]

        self.window_idx = window_idx
        self.include_final_window = include_final_window
        self.approve_sparsity = approve_sparsity

        assert func_data_type in SUPPORTED_STROLL_TYPES
        self.data_type = func_data_type

        # 0. Standardize the input
        series_list: List[pd.Series] = to_series_list(data)
        self.series_dtype = AttributeParser.determine_type(series_list)
        self.series_key: Tuple[str, ...] = tuple([str(s.name) for s in series_list])

        # 1. Determine the start index
        self.start, self.end = start_idx, end_idx  # type: ignore[var-annotated]
        if self.start is None or self.end is None:
            # We always pass start_idx and end_idx from the FeatureCollection.calculate
            # Hence, this code is only useful for testing purposes
            start, end = _determine_bounds("inner", series_list)

            # update self.start & self.end if it was not passed
            self.start = start if self.start is None else self.start
            self.end = end if self.end is None else self.end

        # Especially useful when the index dtype differs from the win-stride-dtype
        # e.g. -> performing a int-based stroll on time-indexed data
        # Note: this is very niche and thus requires advanced knowledge
        # TODO: this code can be omitted if we remove TimeIndexSampleStridedRolling
        self._update_start_end_indices_to_stroll_type(series_list)

        # 2. Construct the index ranges
        # Either use the passed segment indices or compute the start or end times of the
        # segments. The segment indices have precedence over the stride (and window) for
        # index computation.
        if segment_start_idxs is not None or segment_end_idxs is not None:
            self.strides = None
            if segment_start_idxs is not None and segment_end_idxs is not None:
                # When both the start and end points are passed, the window does not
                # matter.
                self.window = None
                np_start_times = self._parse_segment_idxs(segment_start_idxs)
                np_end_times = self._parse_segment_idxs(segment_end_idxs)
            elif segment_start_idxs is not None:  # segment_end_idxs is None
                np_start_times = self._parse_segment_idxs(segment_start_idxs)
                np_end_times = np_start_times + self._get_np_value(self.window)
            else:  # segment_end_idxs is not None and segment_start_idxs is None
                np_end_times = self._parse_segment_idxs(segment_end_idxs)  # type: ignore[arg-type]
                np_start_times = np_end_times - self._get_np_value(self.window)
        else:
            np_start_times = self._construct_start_idxs()
            np_end_times = np_start_times + self._get_np_value(self.window)

        # Check the numpy start and end indices
        _check_start_end_array(np_start_times, np_end_times)

        # 3. Create a new-index which will be used for DataFrame reconstruction
        # Note: the index-name of the first passed series will be re-used as index-name
        self.index = self._get_output_index(
            np_start_times, np_end_times, name=series_list[0].index.name
        )

        # 4. Store the series containers
        self.series_containers = self._construct_series_containers(
            series_list, np_start_times, np_end_times
        )

        # 5. Check the sparsity assumption
        if not self.approve_sparsity and len(self.index):
            for container in self.series_containers:
                # Warn when min != max
                if np.ptp(container.end_indexes - container.start_indexes) != 0:
                    warnings.warn(
                        f"There are gaps in the sequence of the {container.name}"
                        f"-series!",
                        RuntimeWarning,
                    )

    def _calc_nb_segments_for_stride(self, stride: T) -> int:
        """Calculate the number of output items (segments) for a given single stride."""
        assert self.start is not None and self.end is not None  # for mypy
        nb_feats = max((self.end - self.start - self.window) // stride + 1, 0)
        # Add 1 if there is still some data after (including) the last window its
        # start index - this is only added when `include_last_window` is True.
        nb_feats += self.include_final_window * (
            self.start + stride * nb_feats <= self.end
        )
        return nb_feats

    def _get_np_start_idx_for_stride(self, stride: T) -> np.ndarray:
        """Compute the start index for the given single stride."""
        # ---------- Efficient numpy code -------
        np_start = self._get_np_value(self.start)
        np_stride = self._get_np_value(stride)
        # Compute the start times (these remain the same for each series)
        return np.arange(
            start=np_start,
            stop=np_start + self._calc_nb_segments_for_stride(stride) * np_stride,
            step=np_stride,
        )

    def _construct_start_idxs(self) -> np.ndarray:
        """Construct the start indices of the segments (for all stride values).

        To realize this, we compute the start idxs for each stride and then merge them
        together (without duplicates) in a sorted array.
        """
        start_idxs = []
        for stride in self.strides:
            start_idxs += [self._get_np_start_idx_for_stride(stride)]
        # note - np.unique also sorts the array
        return np.unique(np.concatenate(start_idxs))

    def _get_output_index(
        self, start_idxs: np.ndarray, end_idxs: np.ndarray, name: str
    ) -> pd.Index:
        """Construct the output index."""
        if self.window_idx == "end":
            return pd.Index(end_idxs, name=name)
        elif self.window_idx == "middle":
            return pd.Index(
                start_idxs + ((end_idxs - start_idxs) / 2),
                name=name,
            )
        elif self.window_idx == "begin":
            return pd.Index(start_idxs, name=name)
        else:
            raise ValueError(
                f"window index {self.window_idx} must be either of: "
                "['end', 'middle', 'begin']"
            )

    def _construct_series_containers(
        self,
        series_list: List[pd.Series],
        np_start_times: np.ndarray,
        np_end_times: np.ndarray,
    ) -> List[StridedRolling._NumpySeriesContainer]:
        series_containers: List[StridedRolling._NumpySeriesContainer] = []
        for series in series_list:
            if not self.reset_series_index_b4_segmenting:
                np_idx_times = series.index.values
            else:
                np_idx_times = np.arange(len(series))
                # note: using pd.RangeIndex instead of arange gives the same performance

            series_name = series.name
            if self.data_type is np.ndarray:  # FuncWrapper.input_type is np.ndarray
                # create a non-writeable view of the series
                series = series.values  # np.array will be stored in the SeriesContainer
                series.flags.writeable = False
            elif self.data_type is pd.Series:  # FuncWrapper.input_type is pd.Series
                # pd.Series will be stored in the SeriesContainer
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
                `<series_col_name(s)>_<feature_name>__w=<window>`.

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

        t_start = time.perf_counter()

        # --- Future work ---
        # would be nice if we could optimize this double for loop with something
        # more vectorized
        #
        # As for now we use a map to apply the function (as this evaluates its
        # expression only once, whereas a list comprehension evaluates its expression
        # every time).
        # See more why: https://stackoverflow.com/a/59838723
        out: np.ndarray
        if func.vectorized:
            # Vectorized function execution

            ## IMPL 1
            ## Results in a high memory peak as a new np.array is created (and thus no
            ## view is being used)
            # out = np.asarray(
            #         func(
            #             *[
            #                 np.array([
            #                     sc.values[sc.start_indexes[idx]: sc.end_indexes[idx]]
            #                     for idx in range(len(self.index))
            #                 ])
            #                 for sc in self.series_containers
            #             ],
            #         )
            #     )

            ## IMPL 2
            ## Is a good implementation (equivalent to the one below), will also fail in
            ## the same cases, but it does not perform clear assertions (with their
            ## accompanied clear messages).
            # out = np.asarray(
            #     func(
            #         *[
            #             _sliding_strided_window_1d(sc.values, self.window, self.stride)
            #             for sc in self.series_containers
            #         ],
            #     )
            # )

            views: List[np.ndarray] = []
            for sc in self.series_containers:
                if len(sc.start_indexes) == 0:
                    # There are no feature windows  -> return empty array (see below)
                    views = []
                    break
                elif len(sc.start_indexes) == 1:
                    # There is only 1 feature window (bc no steps in the sliding window)
                    views.append(
                        np.expand_dims(
                            sc.values[sc.start_indexes[0] : sc.end_indexes[0]],
                            axis=0,
                        )
                    )
                else:
                    # There are >1 feature windows (bc >=1 steps in the sliding window)
                    windows = sc.end_indexes - sc.start_indexes
                    strides = sc.start_indexes[1:] - sc.start_indexes[:-1]
                    assert np.all(windows == windows[0]), (
                        "Vectorized functions require same number of samples in each "
                        + "segmented window!"
                    )
                    assert np.all(
                        strides == strides[0]
                    ), "Vectorized functions require same number of samples as stride!"
                    views.append(
                        _sliding_strided_window_1d(
                            sc.values[sc.start_indexes[0] :],
                            windows[0],
                            strides[0],
                            len(self.index),
                        )
                    )

            # Assign empty array as output when there is no view to apply the vectorized
            # function on (this is the case when there is at least for one series no
            # feature windows)
            out = func(*views) if len(views) >= 1 else np.array([])

            out_type = type(out)
            out = np.asarray(out)
            # When multiple outputs are returned (= tuple) they should be transposed
            # when combining into an array
            out = out.T if out_type is tuple else out

        else:
            # Function execution over slices (default)
            out = np.array(
                list(
                    map(
                        func,
                        *[
                            [
                                sc.values[sc.start_indexes[idx] : sc.end_indexes[idx]]
                                for idx in range(len(self.index))
                            ]
                            for sc in self.series_containers
                        ],
                    )
                )
            )

        # Check if the function output is valid.
        # This assertion will be raised when e.g. np.max is applied vectorized without
        # specifying axis=1.
        assert out.ndim > 0, "Vectorized function returned only 1 (non-array) value!"

        # Aggregate function output in a dictionary
        output_names = list(map(self._create_feat_col_name, feat_names))
        feat_out = _process_func_output(out, self.index, output_names, str(func))
        # Log the function execution time
        log_strides = (
            "manual" if self.strides is None else tuple(map(str, self.strides))
        )
        log_window = "manual" if self.window is None else self.window
        _log_func_execution(
            t_start, func, self.series_key, log_window, log_strides, output_names  # type: ignore[arg-type]
        )

        return pd.DataFrame(feat_out, index=self.index)

    # --------------------------------- STATIC METHODS ---------------------------------
    @staticmethod
    def _get_np_value(val: Union[np.number, pd.Timestamp, pd.Timedelta]) -> np.number:
        # Convert everything to int64
        if isinstance(val, pd.Timestamp):
            return val.to_datetime64()
        elif isinstance(val, pd.Timedelta):
            return val.to_timedelta64()
        else:
            return val

    @staticmethod
    def construct_output_index(
        series_keys: Union[str, Tuple[str, ...]], feat_name: str, win_str: str
    ) -> str:
        series_keys = to_tuple(series_keys)
        return f"{'|'.join(series_keys)}__{feat_name}__w={win_str}"

    # ----------------------------- OVERRIDE THESE METHODS -----------------------------
    @abstractmethod
    def _update_start_end_indices_to_stroll_type(
        self, series_list: List[pd.Series]
    ) -> None:
        # NOTE: This method will only be implemented (with code != pass) in the
        # TimeIndexSampleStridedRolling
        raise NotImplementedError

    @abstractmethod
    def _parse_segment_idxs(self, segment_idxs: np.ndarray) -> np.ndarray:
        """Check the segment indexes array to lie between self.start and self.end and
        convert it to the correct dtype (if necessary)."""
        raise NotImplementedError

    @abstractmethod
    def _create_feat_col_name(self, feat_name: str) -> str:
        raise NotImplementedError


class SequenceStridedRolling(StridedRolling):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: Union[int, float],
        strides: Optional[Union[int, float, List[int], List[float]]] = None,
        *args,
        **kwargs,
    ):
        # Set the data type & call the super constructor
        self.win_str_type = DataType.SEQUENCE
        super().__init__(data, window, strides, *args, **kwargs)

    # ------------------------------- Overridden methods -------------------------------
    def _update_start_end_indices_to_stroll_type(
        self, series_list: List[pd.Series]
    ) -> None:
        pass

    def _parse_segment_idxs(self, segment_idxs: np.ndarray) -> np.ndarray:
        if any((segment_idxs < self.start) | (segment_idxs > self.end)):
            warnings.warn(self.OUTSIDE_DATA_BOUNDS_WARNING, RuntimeWarning)
        return segment_idxs

    def _create_feat_col_name(self, feat_name: str) -> str:
        if self.window is not None:
            win_str = str(self.window)
        else:
            win_str = "manual"
        return self.construct_output_index(
            series_keys=self.series_key, feat_name=feat_name, win_str=win_str
        )


class TimeStridedRolling(StridedRolling):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: pd.Timedelta,
        strides: Optional[Union[pd.Timedelta, List[pd.Timedelta]]] = None,
        *args,
        **kwargs,
    ):
        # Check that each series / dataframe has the same tz
        data = to_series_list(data)
        tz_index = data[0].index.tz
        for data_entry in to_series_list(data)[1:]:
            assert (
                data_entry.index.tz == tz_index
            ), "strided rolling input data must all have same timezone"
        self._tz_index = tz_index
        # Set the data type & call the super constructor
        self.win_str_type = DataType.TIME
        super().__init__(data, window, strides, *args, **kwargs)

    # -------------------------------- Extended methods --------------------------------
    def _get_output_index(
        self, start_idxs: np.ndarray, end_idxs: np.ndarray, name: str
    ) -> pd.Index:
        assert start_idxs.dtype.type == np.datetime64
        assert end_idxs.dtype.type == np.datetime64
        if not len(start_idxs):  # to fix "datetime64 values must have a unit specified"
            assert not len(end_idxs)
            start_idxs = start_idxs.astype("datetime64[ns]")
            end_idxs = end_idxs.astype("datetime64[ns]")
        start_idxs = pd.to_datetime(start_idxs, utc=True).tz_convert(self._tz_index)
        end_idxs = pd.to_datetime(end_idxs, utc=True).tz_convert(self._tz_index)
        return super()._get_output_index(start_idxs, end_idxs, name)

    # ------------------------------- Overridden methods -------------------------------
    def _update_start_end_indices_to_stroll_type(
        self, series_list: List[pd.Series]
    ) -> None:
        pass

    def _parse_segment_idxs(self, segment_idxs: np.ndarray) -> np.ndarray:
        segment_idxs = segment_idxs.astype("datetime64")
        start_, end_ = self.start, self.end
        if start_.tz is not None:
            # Convert to UTC (allowing comparison with the segment_idxs)
            assert end_.tz is not None
            start_ = start_.tz_convert(None)
            end_ = end_.tz_convert(None)
        if any((segment_idxs < start_) | (segment_idxs > end_)):
            warnings.warn(self.OUTSIDE_DATA_BOUNDS_WARNING, RuntimeWarning)
        return segment_idxs

    def _create_feat_col_name(self, feat_name: str) -> str:
        # Convert win to time-string if available :)
        if self.window is not None:
            win_str = timedelta_to_str(self.window)
        else:
            win_str = "manual"
        return self.construct_output_index(
            series_keys=self.series_key, feat_name=feat_name, win_str=win_str
        )


class TimeIndexSampleStridedRolling(SequenceStridedRolling):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        # TODO -> update arguments
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: int,
        strides: Optional[Union[int, List[int]]] = None,
        segment_start_idxs: Optional[np.ndarray] = None,
        segment_end_idxs: Optional[np.ndarray] = None,
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

        .. Note::
            `TimeIndexSampleStridedRolling` **does not support** the
            ``segment_start_idxs`` and ``segment_end_idxs`` arguments. Setting these
            will raise a NotImplementedError.

        """
        if segment_start_idxs is not None or segment_end_idxs is not None:
            raise NotImplementedError(
                "TimeIndexSampleStridedRolling is not implemented to support passing"
                + "segment_start_idxs or segment_end_idxs"
            )

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

        # We retain the first series list to stitch back the output index
        self._series_index = series_list[0].index

        # pass the sliced series list instead of data
        super().__init__(series_list, window, strides, *args, **kwargs)

        assert self.series_dtype == DataType.TIME

        # we want to assure that the window-stride arguments are integers (samples)
        assert all(isinstance(p, int) for p in [self.window] + self.strides)

    def apply_func(self, func: FuncWrapper) -> pd.DataFrame:
        # Apply the function and stitch back the time-index
        df = super().apply_func(func)
        df.index = self._series_index[df.index]
        return df

    # ---------------------------- Overridden methods ------------------------------
    def _update_start_end_indices_to_stroll_type(
        self, series_list: List[pd.Series]
    ) -> None:
        # update the start and end times to the sequence datatype
        self.start, self.end = np.searchsorted(
            series_list[0].index.values,
            [self.start.to_datetime64(), self.end.to_datetime64()],
            "left",
        )


def _sliding_strided_window_1d(
    data: np.ndarray, window: int, step: int, nb_segments: int
) -> np.ndarray:
    """View based sliding strided-window for 1-dimensional data.

    Parameters
    ----------
    data: np.array
        The 1-dimensional series to slide over.
    window: int
        The window size, in number of samples.
    step: int
        The step size (i.e., the stride), in number of samples.
    nb_segments: int
        The number of sliding window steps, this is equal to the number of feature
        windows.

    Returns
    -------
    nd.array
        A view of the sliding strided window of the data.

    """
    # window and step in samples
    assert data.ndim == 1, "data must be 1 dimensional"
    assert isinstance(window, (int, np.integer)), "window must be an integer"
    assert isinstance(step, (int, np.integer)), "step must be an integer"

    assert (step >= 1) & (window < len(data))

    shape = [
        nb_segments,
        window,
    ]

    strides = [
        data.strides[0] * step,
        data.strides[0],
    ]

    return np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides, writeable=False
    )
