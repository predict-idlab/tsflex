"""Utility functions for more convenient feature extraction."""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt"

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .function_wrapper import FuncWrapper, _get_name


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


def _check_start_end_array(start_idxs: np.ndarray, end_idxs: np.ndarray):
    """Check if the start and end indices are valid.

    These are valid if they are of the same length and if the start indices are smaller
    than the end indices.

    Parameters
    ----------
    start_idxs: np.ndarray
        The start indices.
    end_idxs: np.ndarray
        The end indices.
    """
    assert len(start_idxs) == len(
        end_idxs
    ), "start_idxs and end_ixs must have equal length"
    assert np.all(
        start_idxs <= end_idxs
    ), "for all corresponding values: segment_start_idxs <= segment_end_idxs"


def _get_funcwrapper_func_and_kwargs(func: FuncWrapper) -> Tuple[Callable, dict]:
    """Extract the function and keyword arguments from the given FuncWrapper.

    Parameters
    ----------
    func: FuncWrapper
        The FuncWrapper to extract the function and kwargs from.

    Returns
    -------
    Tuple[Callable, dict]
        Tuple of 1st the function of the FuncWrapper (is a Callable) and 2nd the keyword
        arguments of the FuncWrapper.

    """
    assert isinstance(func, FuncWrapper)

    # Extract the function (is a Callable)
    function = func.func

    # Extract the keyword arguments
    func_wrapper_kwargs = dict()
    func_wrapper_kwargs["output_names"] = func.output_names
    func_wrapper_kwargs["input_type"] = func.input_type
    func_wrapper_kwargs["vectorized"] = func.vectorized
    func_wrapper_kwargs.update(func.kwargs)

    return function, func_wrapper_kwargs


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

    func_wrapper_kwargs = {}
    if isinstance(func, FuncWrapper):
        # Extract the function and keyword arguments from the function wrapper
        func, func_wrapper_kwargs = _get_funcwrapper_func_and_kwargs(func)

    output_names = func_wrapper_kwargs.get("output_names")

    def wrap_func(*series: Union[np.ndarray, pd.Series], **kwargs) -> Callable:
        if not passthrough_nans:
            series = [s[~np.isnan(s)] for s in series]
        if any([len(s) < min_nb_samples for s in series]):
            if not isinstance(output_names, list) or len(output_names) == 1:
                return error_val
            return tuple([error_val] * len(output_names))
        return func(*series, **kwargs)

    wrap_func.__name__ = "[robust]__" + _get_name(func)
    if "output_names" not in func_wrapper_kwargs.keys():
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

     More specifically this method does (in the following order):\n
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
        The minimum number of samples that are needed for `func` to be applied
        successfully, by default 1.
        .. Note::
            The number of samples are determined after the `passthrough_nans` filter
            took place.

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
