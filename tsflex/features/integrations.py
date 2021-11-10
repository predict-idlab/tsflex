"""Wrappers for seamless integration of feature functions from other packages."""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt"

import importlib
import pandas as pd
import numpy as np

from typing import Callable, Optional, List, Dict, Union
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


# from tsfresh.feature_extraction.settings import PickeableSettings
# TODO: update this to PicklableSettings, once they approve my PR


def tsfresh_settings_wrapper(settings) -> List[Callable]:
    """Wrapper enabling compatibility with tsfresh feature extraction settings.

    [tsfresh feature extraction settings](https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html)
    is how tsfresh represents a collection of features.<br>

    By using this wrapper, we can plug in the features (that are present in the 
    tsfresh feature extraction settings) in a tsflex ``FeatureCollection``. 
    This enables to easily extract (a collection of) tsfresh features while leveraging
    the flexibility of tsflex.

    Example
    -------

    ```python
    from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
    from tsflex.features.integrations import tsfresh_settings_wrapper
    from tsfresh.feature_extraction import MinimalFCParameters

    minimal_tsfresh_feats = MultipleFeatureDescriptors(
        functions=tsfresh_settings_wrapper(MinimalFCParameters()),
        series_names=["sig_0", "sig_1"],  # list of signal names
        windows="15min", strides="2min",
    )

    fc = FeatureCollection(minimal_tsfresh_feats)
    fc.calculate(data)  # calculate the features on your data
    ```

    Parameters
    ----------
    settings: PicklableSettings
        The tsfresh base object for feature settings (which is a dict).

    Returns
    -------
    List[Callable]
        List of the (wrapped) tsfresh functions that are now directly compatible with
        with tsflex.
 
    """
    functions = []
    tsfresh_mod = importlib.import_module("tsfresh.feature_extraction.feature_calculators") 
    for func_name, param in settings.items():
        func = getattr(tsfresh_mod, func_name) 
        if param is None:
            functions.append(func)
        elif getattr(func, "fctype") == "combiner":
            functions.append(tsfresh_combiner_wrapper(func, param))
        else:
            for kwargs in param:
                functions.append(FuncWrapper(func, output_names=f"{func.__name__}_{str(kwargs)}", **kwargs))
    return functions
