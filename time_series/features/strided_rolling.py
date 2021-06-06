"""Contains a (rather) fast implementation of a strided rolling window."""

__author__ = "Vic Degraeve, Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import time
from typing import Callable, Union, Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from ..utils.timedelta import tightest_timedelta_bounds
from .function_wrapper import NumpyFuncWrapper
from .logger import logger


class StridedRolling:
    """Custom sliding window with stride for pandas DataFrames."""

    # Only keep pd.Series and not Union
    def __init__(
        self,
        df: Union[pd.Series, pd.DataFrame],
        window: Union[int, pd.Timedelta],
        stride: Union[int, pd.Timedelta]
    ):
        """Create StridedRolling object.

        Parameters
        ----------
        df : Union[pd.Series, pd.DataFrame]
            :class:`pd.Series` or :class:`pd.DataFrame` to slide over, the index must
            be a (time-zone-aware) :class:`pd.DatetimeIndex`.
        window : Union[int, pd.Timedelta]
            Either an int or `pd.Timedelta`, representing the sliding window length in
            samples or the sliding window duration, respectively.
        stride : Union[int, pd.Timedelta]
            Either an int or `pd.Timedelta`, representing the stride size in samples or
            the stride duration, respectively.

        Note
        ----
        Time based window-stride parameters will be converted into integers at inference
        time. The integer conversion will use flooring, and can be written as the
        following formula::

            time_parameter // series_period

        """
        # TODO: needs some cleanup (remove comments)
        # construct the (expanded) sliding window-stride array
        # Old code: self.time_indexes = df.index[:-window + 1][::stride]
        # Index indicates the start of the windows
        # df = df.to_frame() if isinstance(df, pd.Series) else df # TODO

        # store the orig input (can be pd.Timedelta)
        self.orig_window = window
        self.orig_stride = stride

        self.window: int = StridedRolling._time_arg_to_int(window, df)
        self.stride: int = StridedRolling._time_arg_to_int(stride, df)

        # Index indicates the end of the windows
        self.time_indexes = df.index[self.window - 1:][::self.stride]

        # TODO: Make this here lazy by only doing on first call of apply func
        #   with the goal to lower memory usage (peak)
        self._key = df.name if isinstance(df, pd.Series) else tuple(df.columns.values)
        self._val = data_expansion(df, window=self.window, stride=self.stride)
        # self._strided_vals = {}
        # for col in df.columns:
        #     self._strided_vals[col] = sliding_window(
        #         df[col], window=self.window, stride=self.stride
        #     )

    @staticmethod
    def _time_arg_to_int(
            arg: Union[int, pd.Timedelta],
            df: Union[pd.DataFrame, pd.Series]
    ) -> int:
        """Convert the time arg into a int and **uses flooring**.

        Parameters
        ----------
        arg: Union[int, pd.Timedelta]
            The possible time-based arg that will be converted into an string
        df:
            The `pd.DatetimeIndex`ed dataframe that will be used to infer the frequency
            if necessary.

        Returns
        -------
        int
            The converted int

        Raises
        ------
        ValueError
            When the frequency could not be inferred from the `pd.DatetimeIndex`.
        """
        if isinstance(arg, int):
            return arg
        elif isinstance(arg, pd.Timedelta):
            # use the df to determine the freq
            freq: str = pd.infer_freq(df.index)
            if freq is None:
                raise ValueError(f'could not infer frequency from df {df.columns}')
            try:
                # https://stackoverflow.com/a/31471631/9010039
                int_arg = arg // pd.to_timedelta(to_offset(freq))
                return int_arg
            except Exception:
                print("arg:", arg, "\tfreq: ", freq)
        raise ValueError(f"arg {arg} has invalid type = {type(arg)}")

    # @property
    # def strided_vals(self) -> Dict[str, np.ndarray]:
    #     """Get the expanded series or dataframe.

    #     Returns:
    #     -------
    #     Dict[str, np.ndarray]
    #         A `dict` with the column-name as key, and the corresponding expanded
    #         series as value.

    #     """
    #     return self._vals

    def apply_func(self, np_func: Union[Callable, NumpyFuncWrapper], single_series_func: Optional[bool] = True) -> pd.DataFrame:
        """Apply a function to the expanded time-series.

        Note
        ----
        * If `np_func` is only a callable argument, with no additional logic, this
            will only work for a one-to-one mapping, i.e., no multiple feature-output
            columns are supported for this case!
        * If you want to calculate one-to-many -> `np_func` should be
             a `NumpyFuncWrapper` instance and explicitly use
             the `output_names` attributes of its constructor.

        Parameters
        ----------
        np_func : Union[Callable, NumpyFuncWrapper]
            The Callable (wrapped) function which will be applied.
        single

        Returns
        -------
        pd.DataFrame
            The merged output of the function applied to every column in a
            new DataFrame. The DataFrame's column-names have the format:
                `<signal_col_name>_<feature_name>__w=<window>_s=<stride>`.

        """
        # Convert win & stride to time-string if available :)
        def create_feat_col_name(feat_name) -> str:
            win_str, stride_str = str(self.window), str(self.stride)
            if isinstance(self.orig_window, pd.Timedelta):
                win_str = tightest_timedelta_bounds(self.orig_window)
            if isinstance(self.orig_stride, pd.Timedelta):
                stride_str = tightest_timedelta_bounds(self.orig_stride)
            win_stride_str = f"w={win_str}_s={stride_str}"
            return f"{self._key}_{feat_name}__{win_stride_str}"

        feat_out = {}
        if not isinstance(np_func, NumpyFuncWrapper):
            np_func = NumpyFuncWrapper(np_func)
        feat_names = np_func.output_names

        t_start = time.time()

        if single_series_func:
            # Single-series func
            assert isinstance(self._key, str)
            out = np.apply_along_axis(np_func, axis=-1, arr=self._val)
        else:
            # Multi-series func
            assert isinstance(self._key, tuple) & (len(self._key) > 1)
            nb_inputs = len(self._key)
            def multiseries__val_along_axis(func):
                def wrapper(arr, **kwargs):
                    return func(*[arr[self.window*idx:self.window*(idx+1)] for idx in range(nb_inputs)], **kwargs)
                return wrapper
            out = np.apply_along_axis(multiseries__val_along_axis(np_func), axis=-1, arr=self._val)

        if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
            assert len(feat_names) == 1
            feat_out[create_feat_col_name(feat_names[0])] = out.flatten()
        if out.ndim == 2 and out.shape[1] > 1:
            assert len(feat_names) == out.shape[1]
            for col_idx in range(out.shape[1]):
                feat_out[create_feat_col_name(feat_names[col_idx])] = out[:, col_idx]

        # for sig_key in self.strided_vals.keys(): # TODO heeft enkel meerdere keys als multi series func is??
        #     out = np.apply_along_axis(np_func, axis=-1, arr=self.strided_vals[sig_key])
        #     if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
        #         assert len(feat_names) == 1
        #         feat_out[create_feat_col_name(sig_key, feat_names[0])] = out.flatten()
        #     if out.ndim == 2 and out.shape[1] > 1:
        #         assert len(feat_names) == out.shape[1]
        #         for col_idx in range(out.shape[1]):
        #             feat_out[
        #                 create_feat_col_name(sig_key, feat_names[col_idx])
        #             ] = out[:, col_idx]

        elapsed = time.time() - t_start
        logger.info(
            f"Finished function [{np_func.func.__name__}] on "
            f"{[self._key]} with window-stride "
            f"[{self.window}, {self.stride}] in [{elapsed} seconds]!"
        )

        return pd.DataFrame(index=self.time_indexes, data=feat_out)


def data_expansion(df: Union[pd.Series,pd.DataFrame], window: int, stride=1) -> np.ndarray:
    """Calculate a strided sliding window over a series or dataframe.

    Note
    ----
    When using this function on a series, the default value for axis is fine.
    But, when using this function on a dataframe, you should use axis=0.

    Parameters
    ----------
    df : Union[pd.Series,pd.DataFrame]
        :class:`pd.Series` or :class:`pd.DataFrame` to slide over.
    window : int
        Sliding window length in samples.
    stride : int, optional
        Step/stride length in samples, by default 1.

    Returns
    -------
    np.ndarray
        The expanded series or dataframe, a 2D array of shape:
        (len(`series`)//`stride`, `window`) in case of a series
        (len(`series`)//`stride`, `window` * len(`df.columns`)) in case of a dataframe

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
    if not isinstance(df, pd.Series):
        assert df.shape[1] > 1
        single_series = False
    axis = -1 if single_series else 0
    # TODO: het werkt op DataFrame als je axis = 0 -> wrapper code errond
    data = df.values
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")
    if stride < 1:
        raise ValueError("Step size may not be zero or negative")
    if window > data.shape[axis]:
        info_str = ""
        if isinstance(data, pd.Series):
            info_str += f"Series {df.name} {df.shape}"
        else:
            info_str += f"DataFrame {df.columns.values} {df.shape}"
        print(info_str, " -  window:", window, ", stride:", stride)
        print("Series", df.index.to_series().diff().value_counts())
        raise ValueError("Sliding window size may not exceed size of selected axis")

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stride - window / stride + 1).astype(int)
    shape.append(window)

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
    strides = list(data.strides)
    strides[axis] *= stride
    strides.append(data.strides[axis])

    if single_series:
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).reshape(-1, window*len(df.columns))
