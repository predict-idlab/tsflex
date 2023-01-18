"""Tests for the processing utilities."""

from typing import List

import pandas as pd

from tsflex.chunking import chunk_data
from tsflex.processing import SeriesPipeline, SeriesProcessor, dataframe_func
from tsflex.processing.utils import process_chunks_multithreaded

from .utils import dummy_data


def test_process_chunks_multithreaded(dummy_data):
    def interpolate(series: pd.Series) -> pd.Series:
        return series.interpolate()

    @dataframe_func  # Note: this can (and should) be a single_series_func as dropna works on pd.Series
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    inp = dummy_data.copy()
    inp.loc[inp["TMP"] > 31.5, "TMP"] = pd.NA
    assert any(inp["TMP"].isna())  # Check that there are some NANs present
    series_pipeline = SeriesPipeline(
        [
            SeriesProcessor(series_names="TMP", function=interpolate),
            SeriesProcessor(series_names="TMP", function=drop_nans),
        ]
    )

    out: List[pd.DataFrame] = process_chunks_multithreaded(
        same_range_chunks_list=chunk_data(
            data=dummy_data,
            fs_dict={"EDA": 4, "TMP": 4, "ACC_x": 4, "ACC_y": 4, "ACC_z": 4},
        ),
        series_pipeline=series_pipeline,
        n_jobs=None,
        show_progress=True,
        return_df=True,
        return_all_series=True,
    )

    # we do not have sub-chunks / gaps -> so len out must be 0
    assert len(out) == 1
    assert isinstance(out[0], pd.DataFrame)
    assert set(out[0].columns) == set(dummy_data.columns)
