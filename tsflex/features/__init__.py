"""Feature extraction submodule.

.. include:: ../../docs/pdoc_include/features.md

"""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

from .. import __pdoc__
from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .feature_collection import FeatureCollection
from .feature_collection_sk import SKFeatureCollection
from .function_wrapper import NumpyFuncWrapper
from .logger import get_feature_logs, get_function_stats, get_key_stats

__pdoc__["NumpyFuncWrapper.__call__"] = True

__all__ = [
    "FeatureDescriptor",
    "MultipleFeatureDescriptors",
    "FeatureCollection",
    "SKFeatureCollection",
    "NumpyFuncWrapper",
    "get_feature_logs",
    "get_function_stats",
    "get_key_stats",
]
