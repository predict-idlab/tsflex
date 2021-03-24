"""Fixtures and helper functions for testing."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pytest
import pandas as pd

from typing import Dict


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    df = pd.read_feather("../data/physio.feather")
    df.set_index("timestamp", inplace=True)
    return df


def dataframe_to_series_dict(df: pd.DataFrame) -> Dict[str, pd.Series]:
    output_dict = dict()
    for col in df.columns:
        output_dict[col] = df[col]
    return output_dict


def series_to_series_dict(series: pd.Series) -> Dict[str, pd.Series]:
    assert series.name is not None, "Series must have a name in order to get a key!"
    return {series.name: series}


# def check_eq_na(s1: pd.Series, s2: pd.Series) -> bool:
#     """Checks whether given series are equal while being robust against NANs."""
#     return all((s1 ==  s1) | (s1.isna() &  s2.isna()))
