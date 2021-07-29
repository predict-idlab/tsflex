"""Wrappers for seamless integration of feature functions from other packages."""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt"

import pandas as pd
import numpy as np

from typing import Callable, Any, Optional, List, Dict, Union
from .feature import FuncWrapper


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


def make_robust(
    func: Callable,
    min_nb_samples: int = 1,
    error_val: Any = np.nan,
    output_names: Optional[Union[str, List[str]]]= None,
    passthrough_nans: bool = True,
    **kwargs,
) -> FuncWrapper:
    """Decorates `func` into a robust funcwrapper.

    More specifically this method does:<br>
    * `np.NaN` data input propagation / filtering
    *  `min_nb_samples` checking before feeding to `func`
       (if not met, returns `error_val`)

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
       by default True
   **kwargs:
       Additional keyword arguments

   Returns
   -------
   FuncWrapper
       The robust funcwrapper

   """
    def wrap_func(x: np.array):
        # todo -> fix multiple arrays/series as input
        if not passthrough_nans:
            x = x[~x.isnan()]
        if len(x) < min_nb_samples:
            return error_val
        return func(x)

    wrap_func.__name__ = "[robust]__" + _get_name(func)
    output_names = _get_name(func) if output_names is None else output_names
    return FuncWrapper(wrap_func, output_names=output_names, **kwargs)


# ------------------------------------- SEGLEARN -------------------------------------
def seglearn_wrapper(func: Callable, output_names: Optional[str] = None) -> FuncWrapper:
    """Wrapper enabling compatibility with seglearn functions.
    
    As [seglearn feature-functions](https://github.com/dmbee/seglearn/blob/master/seglearn/feature_functions.py) 
    are vectorized along the first axis (axis=0), we need to expand our window-data.  
    This wrapper converts `1D np.array` to a `2D np.array` with all the window-data in 
    `axis=1`.

    Parameters
    ----------
    func: Callable
        The seglearn function.
    output_names: str, optional
        The output name for the function its output.

    Returns
    -------
    FuncWrapper
        The wrapped seglearn function that is compatible with tsflex.

    """
    def wrap_func(x: np.ndarray):
        out = func(x.reshape(1, len(x)))
        return out.flatten()

    wrap_func.__name__ = "[seglearn_wrapped]__" + _get_name(func)
    output_names = _get_name(func) if output_names is None else output_names
    return FuncWrapper(wrap_func, output_names=output_names)


# ------------------------------------- TSFRESH -------------------------------------
def tsfresh_combiner_wrapper(func: Callable, param: List[Dict]) -> FuncWrapper:
    """Wrapper enabling compatibility with tsfresh combiner functions.

    [tsfresh feature-funtions](https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_extraction/feature_calculators.py)
    are either of type `simple` or `combiner`.
    * `simple`: feature calculators which calculate a single number  
      **=> integrates natively with tsflex**
    * `combiner`: feature calculates which calculate a bunch of features for a list of parameters. 
       These features are returned as a list of (key, value) pairs for each input parameter.  
       **=> requires wrapping the function to only extract the values of the returned tuples**  

    Parameters
    ----------
    func: Callable
        The tsfresh combiner function.
    param: List[Dict]
        List containing dictionaries with the parameter(s) for the combiner function.
        This is exactly the same ``param`` as you would pass to a tsfresh combiner 
        function.

    Returns
    -------
    FuncWrapper
        The wrapped tsfresh combiner function that is compatible with tsflex.

    """
    def wrap_func(x: np.ndarray):
        out = func(x, param)
        return tuple(t[1] for t in out)

    wrap_func.__name__ = "[tsfresh-combiner_wrapped]__" + _get_name(func)
    input_type = pd.Series if hasattr(func, "index_type") else np.array
    return FuncWrapper(
        wrap_func,
        output_names=[func.__name__ + "_" + str(p) for p in param],
        input_type=input_type,
    )
