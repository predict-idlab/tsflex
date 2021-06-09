"""Contains a (rather) fast implementation of a strided rolling window."""

__author__ = "Vic Degraeve, Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import time
from typing import Callable, Union, Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from ..utils.timedelta import timedelta_to_str
from .function_wrapper import NumpyFuncWrapper
from .logger import logger


class StridedRolling:
    """Custom sliding window with stride for pandas DataFrames."""

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame],
        window: Union[int, pd.Timedelta],
        stride: Union[int, pd.Timedelta]
    ):
        """Create StridedRolling object.

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
        self.orig_window = window
        self.orig_stride = stride

        # Determine the window and stride at runtime
        self.window: int = StridedRolling._time_arg_to_int(window, data)
        self.stride: int = StridedRolling._time_arg_to_int(stride, data)

        # Index indicates the end of the windows
        # TODO: hier emiel zijn code pluggen
        self.time_indexes = data.index[self.window - 1:][::self.stride]
        # Old code: self.time_indexes = data.index[:-window + 1][::stride]

        if isinstance(data, pd.Series):
            self._key = tuple([data.name])
        elif isinstance(data, pd.DataFrame):
            self._key = tuple(data.columns.values)

        # TODO: Make this here lazy by only doing on first call of apply func
        #   with the goal to lower memory usage (peak)
        # Construct the (expanded) sliding window-stride array
        self._expanded_view = StridedRolling._data_expansion(
            data, window=self.window, stride=self.stride
        )

    @staticmethod
    def _time_arg_to_int(
            arg: Union[int, pd.Timedelta],
            data: Union[pd.Series, pd.DataFrame]
    ) -> int:
        """Convert the time arg into a int and **uses flooring**.

        Parameters
        ----------
        arg : Union[int, pd.Timedelta]
            The possible time-based arg that will be converted into an string
        data: Union[pd.Series, pd.DataFrame]
            The `pd.DatetimeIndex`ed series or dataframe that will be used to infer the
            frequency if necessary.

        Returns
        -------
        int
            The converted int

        Raises
        ------
        ValueError
            When the frequency could not be inferred from the `pd.DatetimeIndex`.

        """
        # TODO: Jonas hier bij multiple series-input func fixen
        if isinstance(arg, int):
            return arg
        elif isinstance(arg, pd.Timedelta):
            # use the df to determine the freq
            freq: str = pd.infer_freq(data.index)
            if freq is None:
                raise ValueError(f'could not infer frequency from df {data.columns}')
            try:
                # https://stackoverflow.com/a/31471631/9010039
                int_arg = arg // pd.to_timedelta(to_offset(freq))
                return int_arg
            except Exception:
                print("arg:", arg, "\tfreq: ", freq)
        raise ValueError(f"arg {arg} has invalid type = {type(arg)}")

    @staticmethod
    def _data_expansion(data: Union[pd.Series, pd.DataFrame], window: int, stride=1)\
            -> np.ndarray:
        """Calculate a strided sliding window over a series or dataframe.

        Parameters
        ----------
        data : Union[pd.Series,pd.DataFrame]
            :class:`pd.Series` or :class:`pd.DataFrame` to slide over.
        window : int
            Sliding window length in samples.
        stride : int, optional
            Step/stride length in samples, by default 1.

        Returns
        -------
        np.ndarray
            The expanded series or dataframe, a 2D array of shape:\n
            * (len(`series`)//`stride`, `window`) in case of a series
            * (len(`series`)//`stride`, `window` * len(`df.columns`))
               in case of a dataframe

        Note
        ----
        When using this function on a series, the default value for axis is fine. <br>
        But, when using this function on a dataframe, you should use axis=0.

        Raises
        ------
        ValueError
            If the axis is greater than or equal to the number of data dimensions.
        ValueError
            If the stride is negative.
        ValueError
            If the window is greater than the size of the selected axis

        """
        single_series = True
        if not isinstance(data, pd.Series):
            assert data.shape[1] > 1
            single_series = False
        axis = -1 if single_series else 0
        data_np = data.values
        if axis >= data_np.ndim:
            raise ValueError("Axis value out of range")
        if stride < 1:
            raise ValueError("Step size may not be zero or negative")
        if window > data_np.shape[axis]:
            info_str = ""
            if isinstance(data, pd.Series):
                info_str += f"Series {data.name} {data.shape}"
            else:
                info_str += f"DataFrame {data.columns.values} {data.shape}"
            print(info_str, " -  window:", window, ", stride:", stride)
            print("Series", data.index.to_series().diff().value_counts())
            raise ValueError("Sliding window size may not exceed size of selected axis")

        shape = list(data_np.shape)
        shape[axis] = np.floor(
            data_np.shape[axis] / stride - window / stride + 1).astype(int)
        shape.append(window)

        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
        strides = list(data_np.strides)
        strides[axis] *= stride
        strides.append(data_np.strides[axis])

        # Note: out is **view** of `data_np`
        out = np.lib.stride_tricks.as_strided(
            x=data_np, shape=shape, strides=strides, writeable=False
        )

        if single_series:
            return out

        # Multi series
        # The last operations reshapes the expansion from:
        # [STRIDED-OBSERVATIONS, WINDOW_SIZE, NB_COLUMNS]
        #   -> [STRIDED-OBSERVATIONS, WINDOW_SIZE*NB_COLUMS]
        # Assumptions made:

        # TODO -> test this thoroughly for all use cases
        #    * Various data-types -> strides should return a tuple
        #       (stride for each dimension)
        #    * Think about shapes for differently sampled windows

        # * Same stride (data-size) for all the series' columns -> **works**
        # * Same window for all the series' columns
        # * Same fs for all the series` columns
        return out.reshape(-1, window * len(data.columns))

    def apply_func(self, np_func: Union[Callable, NumpyFuncWrapper],
                   single_series_func: Optional[bool] = True) -> pd.DataFrame:
        """Apply a function to the expanded time-series.

        Parameters
        ----------
        np_func : Union[Callable, NumpyFuncWrapper]
            The Callable (wrapped) function which will be applied.
        single_series_func : bool, optional
            Flag indicating whether the function` is a *single input-series function*
            or a *multiple input-series function*

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
            win_str, stride_str = str(self.window), str(self.stride)
            if isinstance(self.orig_window, pd.Timedelta):
                win_str = timedelta_to_str(self.orig_window)
            if isinstance(self.orig_stride, pd.Timedelta):
                stride_str = timedelta_to_str(self.orig_stride)
            win_stride_str = f"w={win_str}_s={stride_str}"
            return f"{'|'.join(self._key)}__{feat_name}__{win_stride_str}"

        if not isinstance(np_func, NumpyFuncWrapper):
            np_func = NumpyFuncWrapper(np_func)
        feat_names = np_func.output_names

        t_start = time.time()

        if single_series_func:
            # Single input-series func
            assert len(self._key) == 1
            out = np.apply_along_axis(np_func, axis=-1, arr=self._expanded_view)
        else:
            # Multi input-series func
            assert isinstance(self._key, tuple) and (len(self._key) > 1)

            def multi_series__val_along_axis(func):
                # Wrapper to allow `np.apply_along_axis` on multi input-series function
                nb_inputs = len(self._key)

                def wrapper(arr):
                    return func(
                        *[arr[self.window*idx:self.window*(idx+1)]
                          for idx in range(nb_inputs)],
                        )
                return wrapper

            out = np.apply_along_axis(
                func1d=multi_series__val_along_axis(np_func), axis=-1,
                arr=self._expanded_view
            )

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

        return pd.DataFrame(index=self.time_indexes, data=feat_out)
