"""Time Series lib, tool for processing and feature extraction in time series."""
__author__ = "Jonas Van Der Donckt"

from .features.feature import Feature, MultipleFeatures
from .features.feature_collection import FeatureCollection
from .features.function_wrapper import NumpyFuncWrapper

from .processing import (
    SeriesProcessor,
    SeriesProcessorPipeline,
    dataframe_func,
    single_series_func,
)

__all__ = [
    "Feature",
    "MultipleFeatures",
    "FeatureCollection",
    "NumpyFuncWrapper",
    "SeriesProcessor",
    "SeriesProcessorPipeline",
    "dataframe_func",
    "single_series_func",
]
