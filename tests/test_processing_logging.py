"""Tests for the processing functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import os
import pytest
import warnings
import pandas as pd
import numpy as np

from tsflex.processing import dataframe_func
from tsflex.processing import SeriesProcessor, SeriesPipeline
from tsflex.processing.series_pipeline import _ProcessingError
from tsflex.processing import get_processor_logs

from .utils import dummy_data, logging_file_path


test_path = os.path.abspath(os.path.dirname( __file__ ))

def test_simple_processing_logging(dummy_data, logging_file_path):
    def interpolate(series: pd.Series) -> pd.Series:
        return series.interpolate()

    @dataframe_func  # Note: this can (and should) be a single_series_func as dropna works on pd.Series
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    assert not os.path.exists(logging_file_path)

    inp = dummy_data.copy()
    inp.loc[inp["TMP"] > 31.5, "TMP"] = pd.NA
    assert any(inp["TMP"].isna())  # Check that there are some NANs present
    series_pipeline = SeriesPipeline(
        [
            SeriesProcessor(series_names=["TMP", "ACC_x"], function=interpolate),
            SeriesProcessor(series_names="TMP", function=drop_nans),
        ]
    )

    _ = series_pipeline.process(inp, logging_file_path=logging_file_path)
    
    assert os.path.exists(logging_file_path)
    logging_df = get_processor_logs(logging_file_path)

    assert all(logging_df.columns.values == ['log_time', 'function', 'series_names', 'duration'])

    assert len(logging_df) == len(series_pipeline.processing_steps)
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == ['log_time']
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == ['duration']

    assert all(logging_df["function"].values == [step.name for step in series_pipeline.processing_steps])
    assert all(logging_df["series_names"].values == ["(TMP,), (ACC_x,)", "(TMP,)"])


def test_file_warning_processing_logging(dummy_data, logging_file_path):
    def interpolate(series: pd.Series) -> pd.Series:
        return series.interpolate()

    @dataframe_func  # Note: this can (and should) be a single_series_func as dropna works on pd.Series
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    assert not os.path.exists(logging_file_path)

    inp = dummy_data.copy()
    inp.loc[inp["TMP"] > 31.5, "TMP"] = pd.NA
    assert any(inp["TMP"].isna())  # Check that there are some NANs present
    series_pipeline = SeriesPipeline(
        [
            SeriesProcessor(series_names=["TMP", "ACC_x"], function=interpolate),
            SeriesProcessor(series_names="TMP", function=drop_nans),
        ]
    )

    with warnings.catch_warnings(record=True) as w:
        with open(logging_file_path, 'w'):
            pass
        assert os.path.exists(logging_file_path)
        _ = series_pipeline.process(inp, logging_file_path=logging_file_path)
        assert os.path.exists(logging_file_path)
        assert len(w) == 1
        assert all([issubclass(warn.category, RuntimeWarning) for warn in w])
        assert "already exists" in str(w[0])
