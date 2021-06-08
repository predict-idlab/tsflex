"""Time Series lib, tool for processing and feature extraction in time series."""
__docformat__ = 'google'
__author__ = "Jonas Van Der Donckt"

from .features import (
    FeatureDescriptor, 
    MultipleFeatureDescriptors,
    FeatureCollection,
    SKFeatureCollection,
    NumpyFuncWrapper
)

from .processing import (
    SeriesProcessor,
    SeriesPipeline,
    SKSeriesPipeline,
    dataframe_func,
)

from .chunking import chunk_signals

__all__ = [
    "FeatureDescriptor",
    "MultipleFeatureDescriptors",
    "FeatureCollection",
    "SKFeatureCollection",
    "NumpyFuncWrapper",
    "SeriesProcessor",
    "SeriesPipeline",
    "SKSeriesPipeline",
    "dataframe_func",
    "chunk_signals"
]
