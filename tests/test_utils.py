"""Tests for the utils (and the fixtures)."""  # Kinda feels like meta testing

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd

from pandas.testing import assert_index_equal, assert_series_equal
from .utils import dummy_data, dataframe_to_series_dict, series_to_series_dict


def test_load_dummy_data(dummy_data):
    assert isinstance(dummy_data, pd.DataFrame)
    assert dummy_data.shape == (30200, 5)
    assert set(["TMP", "EDA", "ACC_x", "ACC_y", "ACC_z"]) == set(dummy_data.columns)
    assert isinstance(dummy_data["TMP"], pd.Series)
    assert isinstance(dummy_data["EDA"], pd.Series)
    assert isinstance(dummy_data["ACC_x"], pd.Series)
    assert isinstance(dummy_data["ACC_y"], pd.Series)
    assert isinstance(dummy_data["ACC_z"], pd.Series)

def test_to_series_dict(dummy_data):
    # dataframe to series dict
    series_dict = dataframe_to_series_dict(dummy_data)
    assert isinstance(series_dict, dict)
    assert series_dict.keys() == set(dummy_data.columns)
    for key in series_dict.keys():
        assert_index_equal(series_dict[key].index, dummy_data.index)
        assert_series_equal(series_dict[key], dummy_data[key])

    # series to series dict
    series_dict = series_to_series_dict(dummy_data["TMP"])
    assert isinstance(series_dict, dict)
    assert series_dict.keys() == set(["TMP"])
    assert_index_equal(series_dict["TMP"].index, dummy_data.index)
    assert_series_equal(series_dict["TMP"], dummy_data["TMP"])
