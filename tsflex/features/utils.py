"""Utility functions for more convenient feature extraction."""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt"

from typing import Callable, Any, Optional, List, Union, Tuple

import numpy as np
import pandas as pd

from .feature import FuncWrapper


# ---------------------------------- PRIVATE METHODS ----------------------------------

def _determine_bounds(bound_method, series_list: List[pd.Series]) -> Tuple[Any, Any]:
    """Determine the bounds of the passed series.

    Parameters
    ----------
    bound_method: str

    series_list : List[pd.Series]
        The list of series for which the bounds are determined.

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
    elif bound_method == "inner-outer":
        latest_start = series_list[0].index[0]
        latest_stop = series_list[0].index[-1]
        for series in series_list[1:]:
            latest_start = max(latest_start, series.index[0])
            latest_stop = max(latest_stop, series.index[-1])
        return latest_start, latest_stop
    elif bound_method == "outer":
        earliest_start = series_list[0].index[0]
        latest_stop = series_list[0].index[-1]
        for series in series_list[1:]:
            earliest_start = min(earliest_start, series.index[0])
            latest_stop = max(latest_stop, series.index[-1])
        return earliest_start, latest_stop
    else:
        raise ValueError(f"invalid bound method string passed {bound_method}")


def _get_name(func: Callable) -> str:
    """Get the name of the function.

    Parameters
    ----------
    func: Callable
        The function whose name has to be returned, should be either a function or an
        object that is callable.

    Returns
    -------
    str
        The name of ``func`` in case of a function or the name of the class in case
        of a callable object.

    """
    assert callable(func), f"The given argument {func} is not callable!"
    try:
        return func.__name__
    except:
        return type(func).__name__


def _make_single_func_robust(
    func: Union[Callable, FuncWrapper],
    min_nb_samples: int,
    error_val: Any,
    passthrough_nans: bool,
) -> FuncWrapper:
    """Decorate a single`func` into a robust FuncWrapper.

    Parameters
    ----------
    func: Union[Callable, FuncWrapper]
        The function that should be made robust.
    min_nb_samples: int
        The minimum number of samples that are needed for `func` to be applied
        successfully.
    error_val: Any
        The error *return* value if the `min_nb_samples` requirement is not met.
    passthrough_nans: bool
        If set to true, `np.NaN` values, which occur in the data will be passed through.
        Otherwise, the `np.NaN` values will be masked out before being passed to `func`.

    Returns
    -------
    FuncWrapper
        The robust FuncWrapper.

    """
    assert isinstance(func, FuncWrapper) or isinstance(func, Callable)

    # Extract the keyword arguments from the function wrapper
    func_wrapper_kwargs = {}
    if isinstance(func, FuncWrapper):
        _func = func
        func = _func.func
        func_wrapper_kwargs["output_names"] = _func.output_names
        func_wrapper_kwargs["input_type"] = _func.input_type
        func_wrapper_kwargs.update(_func.kwargs)

    output_names = func_wrapper_kwargs.get("output_names")

    def wrap_func(*series: Union[np.ndarray, pd.Series], **kwargs) -> FuncWrapper:
        if not passthrough_nans:
            series = [s[~np.isnan(s)] for s in series]
        if any([len(s) < min_nb_samples for s in series]):
            if not isinstance(output_names, list):
                return error_val
            return tuple([error_val] * len(output_names))
        return func(*series, **kwargs)

    wrap_func.__name__ = "[robust]__" + _get_name(func)
    if not "output_names" in func_wrapper_kwargs.keys():
        func_wrapper_kwargs["output_names"] = _get_name(func)

    return FuncWrapper(wrap_func, **func_wrapper_kwargs)


# ---------------------------------- PUBLIC METHODS -----------------------------------

def make_robust(
    funcs: Union[Callable, FuncWrapper, List[Union[Callable, FuncWrapper]]],
    min_nb_samples: Optional[int] = 1,
    error_val: Optional[Any] = np.nan,
    passthrough_nans: Optional[bool] = True,
) -> Union[FuncWrapper, List[FuncWrapper]]:
    """Decorate `funcs` into one or multiple robust FuncWrappers.

     More specifically this method does:\n
     * `np.NaN` data input propagation / filtering
     *  `min_nb_samples` checking before feeding to `func`
        (if not met, returns `error_val`)\n
     Note: this wrapper is useful for functions that should be robust for empty or
     sparse windows and/or nans in the data.

    Parameters
    ----------
    funcs: Union[Callable, FuncWrapper, List[Union[Callable, FuncWrapper]]]
        The function which will be made robust.
    min_nb_samples: int, optional
        The minimum number of samples that are needed for `func` to be applied, by
        default 1.
        successfully.
    error_val: Any, optional
        The error *return* value if the `min_nb_samples` requirement is not met, by
        default `np.NaN`.
    passthrough_nans: bool, optional
        If set to true, `np.NaN` values, which occur in the data will be passed through.
        Otherwise, the `np.NaN` values will be masked out before being passed to `func`,
        by default True.

    Returns
    -------
    Union[FuncWrapper, List[FuncWrapper]]
        The robust FuncWrapper if a single func is passed or a list of robust
        FuncWrappers when a list of functions is passed.

    """
    if isinstance(funcs, Callable) or isinstance(funcs, FuncWrapper):
        return _make_single_func_robust(
            funcs, min_nb_samples, error_val, passthrough_nans
        )
    return [
        _make_single_func_robust(func, min_nb_samples, error_val, passthrough_nans)
        for func in funcs
    ]
