"""Time Series lib, tool for processing and feature extraction in time series."""
__author__ = "Jonas Van Der Donckt"

from .features.feature import FeatureDescriptor, MultipleFeatureDescriptors
from .features.feature_collection import FeatureCollection
from .features.function_wrapper import NumpyFuncWrapper

from .processing import (
    SeriesProcessor,
    SeriesPipeline,
    dataframe_func,
)

__all__ = [
    "FeatureDescriptor",
    "MultipleFeatureDescriptors",
    "FeatureCollection",
    "NumpyFuncWrapper",
    "SeriesProcessor",
    "SeriesPipeline",
    "dataframe_func",
]
