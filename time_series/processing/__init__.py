"""init.py for processing module."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

from .series_processor import SeriesProcessor
from .series_pipeline import SeriesPipeline
from .series_processor import dataframe_func

__all__ = [
    "dataframe_func",
    "SeriesProcessor",
    "SeriesPipeline",
]
