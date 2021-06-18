"""<i><b>flex</b>ible <b>t</b>ime-<b>s</b>eries operations</i>

.. include:: ../docs/pdoc_include/root_documentation.md
"""

__docformat__ = 'numpy'
__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"
__pdoc__ = {
    # do not show tue utils module
    'tsflex.utils': False,
    # show the seriesprocessor & numpyfuncwrapper their call method
    'SeriesProcessor.__call__': True,
    'NumpyFuncWrapper.__call__': True,
}


# from .processing import (
#     SeriesProcessor,
#     SeriesPipeline,
#     SKSeriesPipeline,
#     dataframe_func,
#     get_processor_logs,
# )
#
# from .features import (
#     FeatureDescriptor,
#     MultipleFeatureDescriptors,
#     FeatureCollection,
#     SKFeatureCollection,
#     NumpyFuncWrapper,
#     get_feature_logs,
# )
#
# from .chunking import chunk_data

__all__ = [
    # # Processing
    # "SeriesProcessor",
    # "dataframe_func",
    # "SeriesPipeline",
    # "SKSeriesPipeline",
    # "get_processor_logs",
    #
    # # Features
    # "NumpyFuncWrapper",
    # "FeatureDescriptor",
    # "MultipleFeatureDescriptors",
    # "FeatureCollection",
    # "SKFeatureCollection",
    # "get_feature_logs",

    # Chunking
    # "chunk_data",

    # documentation
    "__pdoc__"
]
