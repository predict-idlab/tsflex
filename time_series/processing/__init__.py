"""init.py for processing module."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

from .series_processor import SeriesProcessor, dataframe_func
from .series_pipeline import SeriesPipeline

__all__ = [
    "dataframe_func",
    "SeriesProcessor",
    "SeriesPipeline",
]
