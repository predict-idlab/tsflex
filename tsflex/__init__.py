"""Time Series lib, tool for processing and feature extraction in time series."""

__docformat__ = 'numpy'
__author__ = "Jonas Van Der Donckt"

from .features import (
    FeatureDescriptor, 
    MultipleFeatureDescriptors,
    FeatureCollection,
    SKFeatureCollection,
    NumpyFuncWrapper,
    get_feature_logs,
)

from .processing import (
    SeriesProcessor,
    SeriesPipeline,
    SKSeriesPipeline,
    dataframe_func,
    get_processor_logs,
)

from .chunking import chunk_data

__all__ = [
    # Features
    "FeatureDescriptor",
    "MultipleFeatureDescriptors",
    "FeatureCollection",
    "SKFeatureCollection",
    "NumpyFuncWrapper",
    "get_feature_logs",
    # Processing
    "SeriesProcessor",
    "dataframe_func",
    "SeriesPipeline",
    "SKSeriesPipeline",
    "get_processor_logs",
    # Chunking
    "chunk_data",
]
