"""Wrappers for seamless integration of feature functions from other packages."""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt"

import importlib
import pandas as pd
import numpy as np

from typing import Callable, Optional, List, Dict, Union
from .feature import FuncWrapper
from .utils import _get_name


# ------------------------------------- SEGLEARN -------------------------------------
def seglearn_wrapper(func: Callable, func_name: Optional[str] = None) -> FuncWrapper:
    """Wrapper enabling compatibility with seglearn functions.
    
    As [seglearn feature-functions](https://github.com/dmbee/seglearn/blob/master/seglearn/feature_functions.py) 
    are vectorized along the first axis (axis=0), we need to expand our window-data.  
    This wrapper converts `1D np.array` to a `2D np.array` with all the window-data in 
    `axis=1`.

    Parameters
    ----------
    func: Callable
        The seglearn function.
    func_name: str, optional
        The name for the passed function. This will be used when constructing the output
        names.

    Returns
    -------
    FuncWrapper
        The wrapped seglearn function that is compatible with tsflex.

    """
    def wrap_func(x: np.ndarray):
        out = func(x.reshape(1, len(x)))
        return out.flatten()

    wrap_func.__name__ = "[seglearn_wrapped]__" + _get_name(func)
    output_names = _get_name(func) if func_name is None else func_name
    # A bit hacky (hard coded), bc hist is only func that returns multiple values
    if hasattr(func, "bins"):
        output_names = [output_names+f"_bin{idx}" for idx in range(1, func.bins+1)]
    return FuncWrapper(wrap_func, output_names=output_names)


def seglearn_feature_dict_wrapper(features_dict: Dict) -> List[Callable]:
    """Wrapper enabling compatibility with seglearn feature dictionaries.

    seglearn represents a collection of features as a dictionary. 

    By using this wrapper, we can plug in the features (that are present in the 
    dictionary) in a tsflex ``FeatureCollection``. 
    This enables to easily extract (a collection of) seglearn features while leveraging
    the flexibility of tsflex.

    .. Note::
        This wrapper wraps the output of seglearn functions that return feature
        dictionaries;  
        - [base_features()](https://dmbee.github.io/seglearn/feature_functions.html#seglearn.feature_functions.base_features)  
        - [emg_features()](https://dmbee.github.io/seglearn/feature_functions.html#seglearn.feature_functions.emg_features)  
        - [hudgins_features()](https://dmbee.github.io/seglearn/feature_functions.html#seglearn.feature_functions.hudgins_features)  
        - [all_features()](https://dmbee.github.io/seglearn/feature_functions.html#seglearn.feature_functions.all_features)  

    Example
    -------
    ```python
    from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
    from tsflex.features.integrations import seglearn_feature_dict_wrapper
    from seglearn.feature_functions import base_features

    basic_seglearn_feats = MultipleFeatureDescriptors(
        functions=seglearn_feature_dict_wrapper(base_features()),
        series_names=["sig_0", "sig_1"],  # list of signal names
        windows="15min", strides="2min",
    )

    fc = FeatureCollection(basic_seglearn_feats)
    fc.calculate(data)  # calculate the features on your data
    ```

    Parameters
    ----------
    features_dict: Dictionary
        The seglearn collection of features (which is a dict).

    Returns
    -------
    List[Callable]
        List of the (wrapped) seglearn functions that are now directly compatible with
        with tsflex.
 
    """
    return [seglearn_wrapper(func) for func in features_dict.values()]
    

# -------------------------------------- TSFEL --------------------------------------
def tsfel_feature_dict_wrapper(features_dict: Dict) -> List[Callable]:
    """Wrapper enabling compatibility with tsfel feature extraction configurations.

    tsfel represents a collection of features as a dictionary, see more [here](https://tsfel.readthedocs.io/en/latest/descriptions/get_started.html#set-up-the-feature-extraction-config-file).

    By using this wrapper, we can plug in the features (that are present in the 
    tsfel feature extraction configuration) in a tsflex ``FeatureCollection``. 
    This enables to easily extract (a collection of) tsfel features while leveraging
    the flexibility of tsflex.

    .. Note::
        This wrapper wraps the output of tsfel its `get_features_by_domain` or
        `get_features_by_tag`. <br>
        Se more [here](https://github.com/fraunhoferportugal/tsfel/blob/master/tsfel/feature_extraction/features_settings.py).

    Example
    -------
    ```python
    from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
    from tsflex.features.integrations import tsfel_feature_dict_wrapper
    from tsfel.feature_extraction import get_features_by_domain

    stat_tsfel_feats = MultipleFeatureDescriptors(
        functions=tsfel_feature_dict_wrapper(get_features_by_domain("statistical")),
        series_names=["sig_0", "sig_1"],  # list of signal names
        windows="15min", strides="2min",
    )

    fc = FeatureCollection(stat_tsfel_feats)
    fc.calculate(data)  # calculate the features on your data
    ```

    Parameters
    ----------
    features_dict: Dictionary
        The tsfel collection of features (which is a dict).

    Returns
    -------
    List[Callable]
        List of the (wrapped) tsfel functions that are now directly compatible with
        with tsflex.
 
    """
    def get_output_names(config: dict):  
        """Create the output_names based on the configuration."""
        nb_outputs = config["n_features"]
        func_name = config["function"].split(".")[-1]
        if isinstance(nb_outputs, str) and isinstance(config["parameters"][nb_outputs], int):
            nb_outputs = config["parameters"][nb_outputs]
        if func_name == "lpcc":  # Because https://github.com/fraunhoferportugal/tsfel/issues/103
            nb_outputs += 1
        if isinstance(nb_outputs, int):
            if nb_outputs == 1:
                return func_name
            else:
                return [func_name+f"_{idx}" for idx in range(1,nb_outputs+1)]
        output_param = eval(config["parameters"][nb_outputs])
        return [func_name+f"_{nb_outputs}=v" for v in output_param]

    functions = []
    tsfel_mod = importlib.import_module("tsfel.feature_extraction")
    for donain_feats in features_dict.values():  # Iterate over feature domains
        for config in donain_feats.values():  # Iterate over function configs
            func = getattr(tsfel_mod, config["function"].split(".")[-1])
            params = config["parameters"] if config["parameters"] else {}
            output_names = get_output_names(config)
            functions.append(FuncWrapper(func, output_names, **params))
    return functions


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


def tsfresh_settings_wrapper(settings: Dict) -> List[Callable]:
    """Wrapper enabling compatibility with tsfresh feature extraction settings.

    [tsfresh feature extraction settings](https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html)
    is how tsfresh represents a collection of features (as a dict).<br>

    By using this wrapper, we can plug in the features (that are present in the 
    tsfresh feature extraction settings) in a tsflex ``FeatureCollection``. 
    This enables to easily extract (a collection of) tsfresh features while leveraging
    the flexibility of tsflex.

    .. Note::
        This wrapper wraps the output of tsfresh its `MinimalFCParameters()`, 
        `EfficientFCParameters()`, `IndexBasedFCParameters()`, 
        `TimeBasedFCParameters()`, or `ComprehensiveFCParameters()`. <br>
        Se more [here](https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_extraction/settings.py).

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
