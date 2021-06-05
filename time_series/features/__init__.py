"""init.py for features module."""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .feature_collection import FeatureCollection
from .feature_collection_estimator import FeatureCollectionEstimator
from .feature_collection_sk import SKFeatureCollection
from .function_wrapper import NumpyFuncWrapper

__all__ = [
    "FeatureDescriptor",
    "MultipleFeatureDescriptors",
    "FeatureCollection",
    "FeatureCollectionEstimator",
    "SKFeatureCollection",
    "NumpyFuncWrapper",
]
