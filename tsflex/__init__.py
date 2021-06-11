"""tsflex, a flexible tool for time-series first data manipulation."""

__docformat__ = 'numpy'
__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"
__pdoc__ = {
    'tsflex.utils': False,
    # show the seriesprocessor its call method
    'SeriesProcessor.__call__': True,
    # show the numpyfuncwrapper its call method
    'NumpyFuncWrapper.__call__': True,
}


from .processing import (
    SeriesProcessor,
    SeriesPipeline,
    SKSeriesPipeline,
    dataframe_func,
    get_processor_logs,
)

from .features import (
    FeatureDescriptor,
    MultipleFeatureDescriptors,
    FeatureCollection,
    SKFeatureCollection,
    NumpyFuncWrapper,
    get_feature_logs,
)

from .chunking import chunk_data

__all__ = [
    # Processing
    "SeriesProcessor",
    "dataframe_func",
    "SeriesPipeline",
    "SKSeriesPipeline",
    "get_processor_logs",

    # Features
    "NumpyFuncWrapper",
    "FeatureDescriptor",
    "MultipleFeatureDescriptors",
    "FeatureCollection",
    "SKFeatureCollection",
    "get_feature_logs",

    # Chunking
    "chunk_data",
]
