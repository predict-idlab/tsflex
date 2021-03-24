"""Tests for the processing functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pytest
import pandas as pd
import numpy as np

from time_series.processing import single_series_func, dataframe_func
from time_series.processing import SeriesProcessor, SeriesProcessorPipeline

from .utils import dummy_data, dataframe_to_series_dict, series_to_series_dict


## Function wrappers


def test_single_series_func_decorator(dummy_data):
    # Create undecorated single series function
    def to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series)

    # Create decorated single series function
    @single_series_func
    def to_numeric_decorated(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series)

    inp = dummy_data["TMP"].astype(str)
    assert isinstance(inp, pd.Series)

    # Undecorated series function
    single_series_f = to_numeric
    assert not np.issubdtype(inp, np.number)
    res = single_series_f(inp)
    assert isinstance(res, pd.Series)
    assert res.shape == dummy_data["TMP"].shape
    assert np.issubdtype(res, np.number)

    # Decorated series function
    decorated_single_series_f = to_numeric_decorated
    series_dict = series_to_series_dict(inp)
    assert not np.issubdtype(series_dict["TMP"], np.number)
    res = decorated_single_series_f(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.number)


def test_dataframe_func_decorator(dummy_data):
    # Create undecorated dataframe function
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    # Create decorated dataframe function
    @dataframe_func
    def drop_nans_decorated(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    # Insert some NANs into the dummy_data
    assert np.all(~pd.isna(dummy_data))
    dummy_data.iloc[:10] = pd.NA

    # Undecorated datframe function
    dataframe_f = drop_nans
    assert not np.all(~pd.isna(dummy_data))
    res = dataframe_f(dummy_data)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (dummy_data.shape[0] - 10,) + dummy_data.shape[1:]
    assert np.all(~pd.isna(res))

    # Decorated datframe function
    decorated_dataframe_f = drop_nans_decorated
    assert not np.all(~pd.isna(dummy_data))
    series_dict = dataframe_to_series_dict(dummy_data)
    res = decorated_dataframe_f(series_dict)
    assert isinstance(res, pd.DataFrame)  # TODO: WHY NOT RETURN A SERIES DICT?
    assert res.shape == (dummy_data.shape[0] - 10,) + dummy_data.shape[1:]
    assert np.all(~pd.isna(res))


## SeriesProcessor


def test_single_signal_series_processor(dummy_data):
    @single_series_func
    def to_binary(series, thresh_value):
        return series.map(lambda eda: eda > thresh_value)

    series_processor = SeriesProcessor(["EDA"], func=to_binary, thresh_value=0.6)
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert isinstance(res, dict)
    assert res.keys() == set(["EDA"])
    assert isinstance(res["EDA"], pd.Series)
    assert res["EDA"].shape == dummy_data["EDA"].shape

    assert all(res["EDA"][dummy_data["EDA"] <= 0.6] == False)
    assert all(res["EDA"][dummy_data["EDA"] > 0.6] == True)


def test_multi_signal_series_processor(dummy_data):
    @single_series_func
    def percentile_clip(series, l_perc=0.01, h_perc=0.99):
        # Note: this func is useless in ML (data leakage; percentiles are not fitted)
        l_thresh = np.percentile(series, l_perc * 100)
        h_thresh = np.percentile(series, h_perc * 100)
        return series.clip(l_thresh, h_thresh)

    series_processor = SeriesProcessor(["EDA", "TMP"], func=percentile_clip)
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert isinstance(res, dict)
    assert res.keys() == set(dummy_data.columns)
    assert isinstance(res["EDA"], pd.Series)
    assert isinstance(res["TMP"], pd.Series)
    assert res["EDA"].shape == dummy_data["EDA"].shape
    assert res["TMP"].shape == dummy_data["TMP"].shape

    assert min(res["EDA"]) == dummy_data["EDA"].quantile(0.01)
    assert max(res["EDA"]) == dummy_data["EDA"].quantile(0.99)
    assert min(res["TMP"]) == dummy_data["TMP"].quantile(0.01)
    assert max(res["TMP"]) == dummy_data["TMP"].quantile(0.99)


## SeriesProcessorPipeline


def test_single_signal_series_processor_pipeline(dummy_data):
    @single_series_func
    def interpolate(series: pd.Series) -> pd.Series:
        return series.interpolate()

    @dataframe_func
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    inp = dummy_data.copy()
    inp.loc[inp["TMP"] > 31.5, "TMP"] = pd.NA
    assert any(inp["TMP"].isna())
    processing_pipeline = SeriesProcessorPipeline(
        [
            SeriesProcessor(["TMP"], func=interpolate),
            SeriesProcessor(["TMP"], func=drop_nans),
        ]
    )
    res_dict_all = processing_pipeline(inp, return_all_signals=True, return_df=False)
    res_dict_req = processing_pipeline(inp, return_all_signals=False, return_df=False)
    res_df_all = processing_pipeline(inp, return_all_signals=True, return_df=True)
    res_df_req = processing_pipeline(inp, return_all_signals=False, return_df=True)

    assert isinstance(res_dict_all, dict) & isinstance(res_dict_req, dict)
    assert isinstance(res_df_all, pd.DataFrame) & isinstance(res_df_req, pd.DataFrame)
    assert res_dict_all.keys() == set(["TMP", "EDA"])
    assert set(res_df_all.columns) == set(["TMP", "EDA"])
    assert res_dict_req.keys() == set(["TMP"])
    assert res_df_req.columns == ["TMP"]

    assert len(res_dict_all["TMP"]) < len(dummy_data)
    assert len(res_dict_req["TMP"]) < len(dummy_data)
    assert (len(res_df_all) == len(dummy_data)) & (len(res_df_req) < len(dummy_data))

    assert all(~res_df_req["TMP"].isna())
    assert all(~res_dict_all["TMP"].isna()) & all(~res_dict_req["TMP"].isna())
    # NaNs get introduced when merging to df
    assert any(res_df_all["TMP"].isna())

    assert all(res_df_all["TMP"].dropna().values == res_dict_all["TMP"])
    assert all(res_df_req["TMP"].values == res_dict_all["TMP"])

    assert all(res_dict_all["EDA"] == inp["EDA"])
    assert all(res_df_all["EDA"] == inp["EDA"])


def test_multi_signal_series_processor_pipeline(dummy_data):
    @dataframe_func
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    @single_series_func
    def percentile_clip(series, l_perc=0.01, h_perc=0.99):
        # Note: this func is useless in ML (data leakage; percentiles are not fitted)
        l_thresh = np.percentile(series, l_perc * 100)
        h_thresh = np.percentile(series, h_perc * 100)
        return series.clip(l_thresh, h_thresh)

    inp = dummy_data.copy()
    inp.loc[inp["TMP"] > 31.5, "TMP"] = pd.NA
    assert any(inp["TMP"].isna())
    assert all(~inp["EDA"].isna())
    processing_pipeline = SeriesProcessorPipeline(
        [
            SeriesProcessor(["TMP"], func=drop_nans),  # TODO -> on 1 sig fails
            SeriesProcessor(["TMP", "EDA"], func=percentile_clip),
        ]
    )
    res_dict_all = processing_pipeline(inp, return_all_signals=True, return_df=False)
    res_dict_req = processing_pipeline(inp, return_all_signals=False, return_df=False)
    res_df_all = processing_pipeline(inp, return_all_signals=True, return_df=True)
    res_df_req = processing_pipeline(inp, return_all_signals=False, return_df=True)

    assert isinstance(res_dict_all, dict) & isinstance(res_dict_req, dict)
    assert isinstance(res_df_all, pd.DataFrame) & isinstance(res_df_req, pd.DataFrame)
    assert res_dict_all.keys() == set(["TMP", "EDA"])
    assert set(res_df_all.columns) == set(["TMP", "EDA"])
    assert res_dict_req.keys() == set(["TMP", "EDA"])
    assert set(res_df_req.columns) == set(["TMP", "EDA"])

    assert len(res_dict_all["TMP"]) < len(dummy_data)
    assert len(res_dict_req["TMP"]) < len(dummy_data)
    assert (len(res_df_all) == len(dummy_data)) & (len(res_df_req) == len(dummy_data))

    assert all(~res_dict_all["TMP"].isna()) & all(~res_dict_req["TMP"].isna())
    # NaNs get introduced when merging to df
    assert any(res_df_all["TMP"].isna()) & any(res_df_req["TMP"].isna())

    assert all(res_dict_req["TMP"] == res_dict_all["TMP"])
    assert all(res_df_all["TMP"].dropna().values == res_dict_all["TMP"])
    assert all(res_df_req["TMP"].dropna().values == res_dict_all["TMP"])
    assert all(res_dict_req["EDA"] == res_dict_all["EDA"])
    assert all(res_df_all["EDA"] == res_dict_all["EDA"])
    assert all(res_df_req["EDA"] == res_dict_all["EDA"])

    assert min(res_dict_all["EDA"]) == dummy_data["EDA"].quantile(0.01)
    assert max(res_dict_all["EDA"]) == dummy_data["EDA"].quantile(0.99)
    assert min(res_dict_all["TMP"]) == inp["TMP"].quantile(0.01)
    assert max(res_dict_all["TMP"]) == inp["TMP"].quantile(0.99)
