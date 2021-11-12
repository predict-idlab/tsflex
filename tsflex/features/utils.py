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
    elif bound_method == 'inner-outer':
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


# ---------------------------------- PUBLIC METHODS -----------------------------------

def make_robust(
    func: Callable,
    min_nb_samples: int = 1,
    error_val: Any = np.nan,
    output_names: Optional[Union[str, List[str]]] = None,
    passthrough_nans: bool = True,
    **kwargs,
) -> FuncWrapper:
    """Decorates `func` into a robust funcwrapper.

     More specifically this method does:\n
     * `np.NaN` data input propagation / filtering
     *  `min_nb_samples` checking before feeding to `func`
        (if not met, returns `error_val`)\n
     Note: this wrapper is useful for functions that should be robust for empty or sparse
     windows and/or nans in the data.

    Parameters
    ----------
    func: Callable
        The function which will be made robust.
    min_nb_samples: int
        The minimum number of samples that are needed for `func` to be applied
        successfully.
    error_val: Any
        The error *return* value if the `min_nb_samples` requirement is not met.
    output_names: Union[str, List[str]]
        The `func` its output_names
        .. note::
            This value must be set if `func` returns more than 1 output!
    passthrough_nans: bool
        If set to true, `np.NaN` values, which occur in the data will be passed through.
        Otherwise, the `np.NaN` values will be masked out before being passed to `func`,
        by default True.
    **kwargs:
        Additional keyword arguments

    Returns
    -------
    FuncWrapper
        The robust FuncWrapper.

    """

    def wrap_func(*series: Union[np.ndarray, pd.Series], **kwargs) -> FuncWrapper:
        if not passthrough_nans:
            series = [s[~np.isnan(s)] for s in series]
        if any([len(s) < min_nb_samples for s in series]):
            if not isinstance(output_names, list):
                return error_val
            return tuple([error_val] * len(output_names))
        return func(*series, **kwargs)

    wrap_func.__name__ = "[robust]__" + _get_name(func)
    output_names = _get_name(func) if output_names is None else output_names
    return FuncWrapper(wrap_func, output_names=output_names, **kwargs)
