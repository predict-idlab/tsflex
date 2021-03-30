"""Tests for the utils (and the fixtures)."""  # Kinda feels like meta testing

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd

from .utils import dummy_data, dataframe_to_series_dict, series_to_series_dict


def test_load_dummy_data(dummy_data):
    assert isinstance(dummy_data, pd.DataFrame)
    assert dummy_data.shape == (31982, 2)
    assert set(["TMP", "EDA"]) == set(dummy_data.columns)
    assert isinstance(dummy_data["TMP"], pd.Series)
    assert isinstance(dummy_data["EDA"], pd.Series)


def test_to_series_dict(dummy_data):
    # dataframe to series dict
    series_dict = dataframe_to_series_dict(dummy_data)
    assert isinstance(series_dict, dict)
    assert series_dict.keys() == set(dummy_data.columns)
    assert all(series_dict["TMP"] == dummy_data["TMP"])
    assert all(series_dict["EDA"] == dummy_data["EDA"])

    # series to series dict
    series_dict = series_to_series_dict(dummy_data["TMP"])
    assert isinstance(series_dict, dict)
    assert series_dict.keys() == set(["TMP"])
    assert all(series_dict["TMP"] == dummy_data["TMP"])
