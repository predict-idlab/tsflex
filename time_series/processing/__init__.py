"""init.py for processing module."""

__author__ = 'Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt'

from .series_processor import dataframe_func, single_series_func
from .series_processor import SeriesProcessor, SeriesProcessorPipeline

__all__ = [dataframe_func, single_series_func, SeriesProcessor, SeriesProcessorPipeline]
