"""Fixtures and helper functions for testing."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pytest
import pandas as pd

from typing import Dict


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    df1 = pd.read_parquet("../examples/data/empatica/gsr.parquet")
    df2 = pd.read_parquet("../examples/data/empatica/tmp.parquet")
    df3 = pd.read_parquet("../examples/data/empatica/acc.parquet")
    df = pd.merge(df1, df2, how='inner', on='timestamp')
    df = pd.merge(df, df3, how='inner', on='timestamp')
    df.set_index("timestamp", inplace=True)
    return df


def dataframe_to_series_dict(df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {col: df[col].copy() for col in df.columns}


def series_to_series_dict(series: pd.Series) -> Dict[str, pd.Series]:
    assert series.name is not None, "Series must have a name in order to get a key!"
    return {series.name: series.copy()}
