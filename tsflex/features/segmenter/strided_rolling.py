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
    segment_start_idxs: np.ndarray, optional  # TODO is this a numpy array?
        The start indices for the segmented windows. If not provided, the start indices
        will be computed from the data using the passed ``strides``. By default None.
    segment_end_idxs: np.ndarray, optional
        The end indices for the segmented windows. You can only pass an array to this
        argument when you pass an array to `segment_start_idxs` as well (read more in
        note below). If not provided, the end indices will be computed from the data
        using the passed ``strides``. By default None.
        .. Note::
            You can only pass an array for `segment_end_idxs` when an array is passed
            for `segment_start_idxs` that has th following properties;
                - both should have equal length
                - all values in ``segment_start_idxs`` should be <= ``segment_end_idxs``
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
            For a np.array it is possible to create very efficient views, but there is
            no such thing as a pd.Series view. Thus, for each stroll, a new series is
            created.
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
        segment_start_idxs: Optional[np.ndarray] = None,
        segment_end_idxs: Optional[np.ndarray] = None,
        start_idx: Optional[T] = None,
        end_idx: Optional[T] = None,
        func_data_type: Optional[Union[np.array, pd.Series]] = np.array,
        window_idx: Optional[str] = "end",
        include_final_window: bool = False,
        approve_sparsity: Optional[bool] = False,
    ):
        if strides is not None:
            strides = to_list(strides)

        # Check the passed segment indices
        if segment_start_idxs is not None and segment_end_idxs is not None:
            assert len(segment_start_idxs) == len(segment_end_idxs), (
                "The segment_start_idxs and segment_end_idxs should have equal length"
            )
            assert np.all(segment_start_idxs <= segment_end_idxs), (
                "Values in segment_start_idxs should be <= correspend segment_end_idxs value"
            )

        if window is not None:
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
            elif segment_end_idxs is not None:  # segment_start_idxs is None
                np_end_times = self._parse_segment_idxs(segment_end_idxs)
                np_start_times = np_end_times - self._get_np_value(self.window)
        else:
            np_start_times = self._construct_start_idxs()
            np_end_times = np_start_times + self._get_np_value(self.window)
        
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

    def _calc_nb_feats_for_stride(self, stride) -> int:
        """Calculate the number of features for a given stride."""
        nb_feats = max((self.end - self.start - self.window) // stride + 1, 0)
        # Add 1 if there is still some data after (including) the last window its
        # start index - this is only added when `include_last_window` is True.
        nb_feats += self.include_final_window * (
            self.start + stride * nb_feats <= self.end
        )
        return nb_feats

    def _get_np_start_idx_for_stride(self, stride: T) -> np.ndarray:
        """Compute the start index for the given stride."""
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
        """Construct the start indices of the segments (for all stride values).
        
        To realize this, we compute the start idxs for each stride and then merge them
        together (without duplicates) in a sorted array. 
        """
        start_idxs = []
        for stride in self.strides:
            start_idxs += [self._get_np_start_idx_for_stride(stride)]
        # note - np.unique also sorts the array
        return np.unique(np.concatenate(start_idxs))

    def _get_output_index(self, start_idxs: np.ndarray, end_idxs: Union[np.ndarray, None], name: str) -> pd.Index:
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
                    assert np.all(windows == windows[0]), (
                        "Vectorized functions require same number of samples in each "
                        + "segmented window!"
                    )
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
        log_strides = "manual" if self.strides is None else tuple(map(str, self.strides))
        log_window = "manual" if self.window is None else self.window
        logger.info(
            f"Finished function [{func.func.__name__}] on "
            f"{[self.series_key]} with window-stride "
            f"[{log_window}, {log_strides}] in [{elapsed} seconds]!"
        )

        return pd.DataFrame(index=self.index, data=feat_out)

    def _update_start_end_indices_to_stroll_type(self, series_list: List[pd.Series]):
        pass

    # --------------------------------- STATIC METHODS ---------------------------------
    # @staticmethod
    # def calc_nb_features(start, end, window, stride, include_final_window=False) -> int:
    #     nb_feats = max((end - start - window) // stride + 1, 0)
    #     # Add 1 if there is still some data after (including) the last window its
    #     # start index - this is only added when `include_last_window` is True.
    #     nb_feats += include_final_window * (start + stride * nb_feats <= end)
    #     return nb_feats

    @staticmethod
    def _get_np_value(val):
        # Convert everything to int64
        if isinstance(val, pd.Timestamp):
            return val.to_datetime64()
        elif isinstance(val, pd.Timedelta):
            return val.to_timedelta64()
        else:
            return val

    # ----------------------------- OVERRIDE THESE METHODS -----------------------------
    @abstractmethod
    def _parse_segment_idxs(self, segment_idxs: np.ndarray) -> np.ndarray:
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
        # Set the data type & call the super constructor
        self.win_str_type = DataType.SEQUENCE
        super().__init__(data, window, strides, *args, **kwargs)

    # ------------------------------- Overridden methods -------------------------------
    def _parse_segment_idxs(self, segment_idxs: np.ndarray) -> np.ndarray:
        valid_range = (segment_idxs >= self.start) & (segment_idxs < self.end)
        return segment_idxs[valid_range]

    def _create_feat_col_name(self, feat_name: str) -> str:
        # TODO -> this is not that loosely coupled if we want somewhere else in the code
        #        to also reproduce col-name construction
        win_str = "w="
        if self.window is not None:
            win_str += str(self.window)
        else:
            win_str += "manual"
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
        # Check that each series / dataframe has the same tz
        data = to_series_list(data)
        tz_index = data[0].index.tz
        for data_entry in to_series_list(data)[1:]:
            assert data_entry.index.tz == tz_index
        self._tz_index = tz_index
        # Set the data type & call the super constructor
        self.win_str_type = DataType.TIME
        super().__init__(data, window, strides, *args, **kwargs)

    # -------------------------------- Extended methdos --------------------------------
    def _get_output_index(self, start_idxs: np.ndarray, end_idxs: np.ndarray, name: str) -> pd.Index:
        assert start_idxs.dtype.type == np.datetime64
        assert end_idxs.dtype.type == np.datetime64
        start_idxs = pd.to_datetime(start_idxs, utc=True).tz_convert(self._tz_index)
        end_idxs = pd.to_datetime(end_idxs, utc=True).tz_convert(self._tz_index)
        return super()._get_output_index(start_idxs, end_idxs, name)

    # ------------------------------- Overridden methods -------------------------------
    def _parse_segment_idxs(self, segment_idxs: np.ndarray) -> np.ndarray:
        idxs_conv = pd.to_datetime(segment_idxs, utc=True).tz_convert(self._tz_index)
        valid_range = (idxs_conv >= self.start) & (idxs_conv < self.end)
        return segment_idxs[valid_range].astype(np.datetime64)

    # TODO: how bad is it that we don't have freq information anymore?
    # def _construct_output_index(self, series: pd.Series) -> pd.DatetimeIndex:

    def _create_feat_col_name(self, feat_name: str) -> str:
        # Convert win to time-string if available :)
        win_str = "w="
        if self.window is not None:
            win_str += timedelta_to_str(self.window)
        else:
            win_str += "manual"
        return f"{'|'.join(self.series_key)}__{feat_name}__{win_str}"


class TimeIndexSampleStridedRolling(SequenceStridedRolling):
    def __init__(
        self,
        # TODO -> update arguments
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
    def _update_start_end_indices_to_stroll_type(self, series_list: List[pd.Series]):
        # update the start and end times to the sequence datatype
        self.start, self.end = np.searchsorted(
            series_list[0].index.values,
            [self.start.to_datetime64(), self.end.to_datetime64()],
            "left",
        )


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
