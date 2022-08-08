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
from ...utils.data import SUPPORTED_STROLL_TYPES, to_series_list, to_list
from ...utils.attribute_parsing import DataType, AttributeParser
from ...utils.time import timedelta_to_str

# Declare a type variable
T = TypeVar("T")


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
    setpoints: np.ndarray, optional
        The setpoints to use for the sliding window. If not provided, the setpoints
        will be computed from the data using the passed ``strides``. By default None.
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
    include_final_window: bool, optional
        Whether the final (possibly incomplete) window should be included in the
        strided-window segmentation, by default False.

        .. Note::
            The remarks below apply when `include_final_window` is set to True.
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
        strides: Optional[Union[T, List[T]]] = None,
        setpoints: Optional[np.ndarray] = None,
        start_idx: Optional[T] = None,
        end_idx: Optional[T] = None,
        func_data_type: Optional[Union[np.array, pd.Series]] = np.array,
        window_idx: Optional[str] = "end",
        include_final_window: bool = False,
        approve_sparsity: Optional[bool] = False,
    ):
        if strides is not None:
            strides = to_list(strides)

        assert AttributeParser.check_expected_type(
            [window] + ([] if strides is None else strides), self.win_str_type
        )

        self.window = window
        self.strides = strides

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
        self.start, self.end = start_idx, end_idx
        if self.start is None or self.end is None:
            start, end = _determine_bounds("inner", series_list)

            # update self.start & self.end if it was not passed
            self.start = start if self.start is None else self.start
            self.end = end if self.end is None else self.end

        # Especially useful when the index dtype differs from the win-stride-dtype
        # e.g. -> performing a int-based stroll on time-indexed data
        # Note: this is very niche and thus requires advanced knowledge
        # TODO: this code can be omitted if we remove TimeIndexSampleStridedRolling
        self._update_start_end_indices_to_stroll_type(series_list)

        # Either use the passed setpoints or compute the start times of the segments
        # The setpoints have precedence over the stride index computation
        if setpoints is not None:  # use the passed setpoints
            self.strides = None
            segment_start_idxs = setpoints
        else:  # compute the start times of the segments (based on the stride(s))
            segment_start_idxs = self._construct_start_idxs()

        if not len(segment_start_idxs):
            # warnings.warn("No segments found for the given data.", UserWarning)
            # If no segments were found -> set dtype to index dtype
            #   (this avoids ufunc 'add' error from numpy bcs of dtype mismatch)
            segment_start_idxs = segment_start_idxs.astype(series_list[0].index.dtype)

        # 2. Create a new-index which will be used for DataFrame reconstruction
        # Note: the index-name of the first passed series will be re-used as index-name
        self.index = self._get_output_index(series_list[0], segment_start_idxs)

        # 3. Construct the index ranges and store the series containers
        np_start_times = segment_start_idxs
        np_end_times = segment_start_idxs + self._get_np_value(self.window)
        self.series_containers = self._construct_series_containers(
            series_list, np_start_times, np_end_times
        )

        # 4. Check the sparsity assumption
        if not self.approve_sparsity and len(self.index):
            for container in self.series_containers:
                # Warn when min != max
                if np.ptp(container.end_indexes - container.start_indexes) != 0:
                    warnings.warn(
                        f"There are gaps in the sequence of the {container.name}"
                        f"-series!",
                        RuntimeWarning,
                    )

    def _get_np_start_idx_for_stride(self, stride) -> np.ndarray:
        # ---------- Efficient numpy code -------
        np_start = self._get_np_value(self.start)
        np_stride = self._get_np_value(stride)

        # Compute the start times (these remain the same for each series)
        return np.arange(
            start=np_start,
            stop=np_start + self._calc_nb_feats_for_stride(stride) * np_stride,
            step=np_stride,
        )

    def _construct_start_idxs(self) -> np.ndarray:
        start_idxs = []
        for stride in self.strides:
            start_idxs += [self._get_np_start_idx_for_stride(stride)]
        # note - np.unique also sorts the array
        return np.unique(np.concatenate(start_idxs))

    def _calc_nb_feats_for_stride(self, stride) -> int:
        nb_feats = max((self.end - self.start - self.window) // stride + 1, 0)
        # Add 1 if there is still some data after (including) the last window its
        # start index - this is only added when `include_last_window` is True.
        nb_feats += self.include_final_window * (
            self.start + stride * nb_feats <= self.end
        )
        return nb_feats

    def _get_window_offset(self) -> T:
        if self.window_idx == "end":
            return self.window
        elif self.window_idx == "middle":
            return self.window / 2
        elif self.window_idx == "begin":
            if isinstance(self.window, pd.Timedelta):
                return pd.Timedelta(seconds=0)
            return 0
        else:
            raise ValueError(
                f"window index {self.window_idx} must be either of: "
                "['end', 'middle', 'begin']"
            )

    def _get_output_index(self, series: pd.Series, start_idxs: np.ndarray) -> pd.Index:
        if start_idxs.dtype.type == np.datetime64:
            start_idxs = pd.to_datetime(start_idxs, unit="ns", utc=True).tz_convert(
                series.index.tz
            )
        return pd.Index(
            data=start_idxs + self._get_window_offset(),
            name=series.index.name,
            # dtype=series.index.dtype,
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

        t_start = time.time()

        # --- Future work ---
        # would be nice if we could optimize this double for loop with something
        # more vectorized
        #
        # As for now we use a map to apply the function (as this evaluates its
        # expression only once, whereas a list comprehension evaluates its expression
        # every time).
        # See more why: https://stackoverflow.com/a/59838723
        out: np.array = None
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

            views = []
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
                    assert np.all(
                        windows == windows[0]
                    ), "Vectorized functions require same number of samples in each segmented window!"
                    assert np.all(
                        strides == strides[0]
                    ), "Vectorized functions require same number of samples as stride!"
                    views.append(
                        _sliding_strided_window_1d(
                            sc.values, windows[0], strides[0], len(self.index)
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
            # Sequential function execution (default)
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
        feat_out = {}
        if out.ndim == 1 and not len(out):
            # When there are no features calculated (due to no feature windows)
            assert not len(self.index)
            for f_name in feat_names:
                # Will be discarded (bc no index)
                feat_out[self._create_feat_col_name(f_name)] = None
        elif out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
            assert len(feat_names) == 1, f"Func {func} returned more than 1 output!"
            feat_out[self._create_feat_col_name(feat_names[0])] = out.flatten()
        else:
            assert out.ndim == 2 and out.shape[1] > 1
            assert (
                len(feat_names) == out.shape[1]
            ), f"Func {func} returned incorrect number of outputs ({out.shape[1]})!"
            for col_idx in range(out.shape[1]):
                feat_out[self._create_feat_col_name(feat_names[col_idx])] = out[
                    :, col_idx
                ]

        elapsed = time.time() - t_start
        log_strides = (
            tuple() if self.strides is None else tuple(str(s) for s in self.strides)
        )
        logger.info(
            f"Finished function [{func.func.__name__}] on "
            f"{[self.series_key]} with window-stride "
            f"[{self.window}, {log_strides}] in [{elapsed} seconds]!"
        )

        return pd.DataFrame(index=self.index, data=feat_out)

    def _update_start_end_indices_to_stroll_type(self, series_list: List[pd.Series]):
        pass

    # ----------------------------- OVERRIDE THESE METHODS -----------------------------
    @abstractmethod
    def _get_np_value(self, val):
        raise NotImplementedError

    @abstractmethod
    def _create_feat_col_name(self, feat_name: str) -> str:
        raise NotImplementedError


class SequenceStridedRolling(StridedRolling):
    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: float,
        strides: Optional[Union[float, List[float]]] = None,
        *args,
        **kwargs,
    ):
        self.win_str_type = DataType.SEQUENCE
        super().__init__(data, window, strides, *args, **kwargs)

    def _get_np_value(self, val) -> float:
        # The sequence values are already of numpy compatible type
        return val

    def _create_feat_col_name(self, feat_name: str) -> str:
        # TODO -> this is not that loosely coupled if we want somewhere else in the code
        #        to also reproduce col-name construction
        win_str = f"w={self.window}"
        return f"{'|'.join(self.series_key)}__{feat_name}__{win_str}"


class TimeStridedRolling(StridedRolling):
    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: pd.Timedelta,
        strides: Optional[Union[pd.Timedelta, List[pd.Timedelta]]] = None,
        *args,
        **kwargs,
    ):
        self.win_str_type = DataType.TIME
        # TODO: do we check that each series has the same tz?
        super().__init__(data, window, strides, *args, **kwargs)

    # ---------------------------- Overridden methods ------------------------------
    def _get_np_value(self, val):
        # Convert everything to int64
        if isinstance(val, pd.Timestamp):
            return val.to_datetime64()
        elif isinstance(val, pd.Timedelta):
            return val.to_timedelta64()
        raise ValueError(f"Unsupported value type: {type(val)}")

    # TODO: how bad is it that we don't have freq information anymore?
    # def _construct_output_index(self, series: pd.Series) -> pd.DatetimeIndex:
    # ...

    def _create_feat_col_name(self, feat_name: str) -> str:
        # Convert win to time-string if available :)
        win_str = f"w={timedelta_to_str(self.window)}"
        return f"{'|'.join(self.series_key)}__{feat_name}__{win_str}"


class TimeIndexSampleStridedRolling(SequenceStridedRolling):
    # TODO: decide what to do with this class
    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        window: int,
        strides: Optional[Union[int, List[int]]] = None,
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
        super().__init__(series_list, window, strides, *args, **kwargs)

        assert self.series_dtype == DataType.TIME

        # we want to assure that the window-stride arguments are integers (samples)
        assert all(isinstance(p, int) for p in [self.window] + self.strides)

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


def _sliding_strided_window_1d(
    data: np.ndarray, window: int, step: int, nb_segments: int
):
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

    if isinstance(window, float):
        assert window.is_integer(), "window must be an int!"
        window = int(window)
    if isinstance(step, float):
        assert step.is_integer(), "step must be an int!"
        step = int(step)

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
        data, shape=shape, strides=strides  # , writeable=False
    )
