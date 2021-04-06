"""Time Series lib, tool for processing and feature extraction in time series."""
__author__ = "Jonas Van Der Donckt"

from .features.feature import FeatureDescriptor, MultipleFeatureDescriptors
from .features.feature_collection import FeatureCollection
from .features.function_wrapper import NumpyFuncWrapper

from .processing import (
    SeriesProcessor,
    SeriesProcessorPipeline,
    dataframe_func,
    single_series_func,
)

__all__ = [
    "FeatureDescriptor",
    "MultipleFeatureDescriptors",
    "FeatureCollection",
    "NumpyFuncWrapper",
    "SeriesProcessor",
    "SeriesProcessorPipeline",
    "dataframe_func",
    "single_series_func",
]
