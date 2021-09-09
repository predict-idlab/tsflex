"""Wrappers for seamless integration of feature functions from other packages."""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt"

import pandas as pd
import numpy as np

from typing import Callable, Any, Optional, List, Dict, Union
from .feature import FuncWrapper
from .utils import _get_name


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
    are either of type `simple` or `combiner`.\n
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
    def wrap_func(x: Union[np.ndarray, pd.Series]):
        out = func(x, param)
        return tuple(t[1] for t in out)

    wrap_func.__name__ = "[tsfresh-combiner_wrapped]__" + _get_name(func)
    input_type = pd.Series if hasattr(func, "index_type") else np.array
    return FuncWrapper(
        wrap_func,
        output_names=[func.__name__ + "_" + str(p) for p in param],
        input_type=input_type,
    )
