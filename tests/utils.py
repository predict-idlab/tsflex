"""Fixtures and helper functions for testing."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import logging
import os
from typing import Dict

import pandas as pd
import pytest

# Get the project direcory
proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    df1 = pd.read_parquet(proj_dir + "/examples/data/empatica/gsr.parquet")
    df2 = pd.read_parquet(proj_dir + "/examples/data/empatica/tmp.parquet")
    df3 = pd.read_parquet(proj_dir + "/examples/data/empatica/acc.parquet")
    df = pd.merge(df1, df2, how="inner", on="timestamp")
    df = pd.merge(df, df3, how="inner", on="timestamp")
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def dummy_group_data() -> pd.DataFrame:
    df = pd.read_csv(proj_dir + "/examples/data/group_data.csv", index_col=0, header=0)
    return df


@pytest.fixture
def logging_file_path() -> str:
    logging_path = proj_dir + "/tests/logging.log"
    yield logging_path
    # Cleanup after test
    if os.path.exists(logging_path):
        logging.shutdown()
        os.remove(logging_path)


def dataframe_to_series_dict(df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {col: df[col].copy() for col in df.columns}


def series_to_series_dict(series: pd.Series) -> Dict[str, pd.Series]:
    assert series.name is not None, "Series must have a name in order to get a key!"
    return {series.name: series.copy()}
