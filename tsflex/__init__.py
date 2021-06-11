"""tsflex, a flexible tool for time-series first data manipulation."""

__docformat__ = 'numpy'
__author__ = "Jonas Van Der Donckt"

__pdoc__ = {
    'tsflex.utils': False,
    # show the seriesprocessor it's call method
    'SeriesProcessor.__call__': True,
    'processing.SeriesProcessor.__call__': True,
}


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
