"""Tests for the processing functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd
import numpy as np

from time_series.processing import (
    single_series_func,
    dataframe_func,
    numpy_func,
    series_numpy_func,
)
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
    assert not np.any(pd.isna(dummy_data))  # Check that there are no NANs present
    dummy_data.iloc[:10] = pd.NA

    # Undecorated datframe function
    dataframe_f = drop_nans
    assert np.any(pd.isna(dummy_data))  # Check if there are some NANs present
    res = dataframe_f(dummy_data)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (dummy_data.shape[0] - 10,) + dummy_data.shape[1:]
    assert not np.any(pd.isna(res))  # Check that there are no NANs present

    # Decorated datframe function
    decorated_dataframe_f = drop_nans_decorated
    assert np.any(pd.isna(dummy_data))  # Check if there are some NANs present
    series_dict = dataframe_to_series_dict(dummy_data)
    res = decorated_dataframe_f(series_dict)
    assert isinstance(res, pd.DataFrame)  # TODO: WHY NOT RETURN A SERIES DICT?
    assert res.shape == (dummy_data.shape[0] - 10,) + dummy_data.shape[1:]
    assert not np.any(pd.isna(res))  # Check that there are no NANs present


def test_numpy_func_decorator(dummy_data):
    # Create undecorated numpy function
    def numpy_is_close_med(sig: np.ndarray) -> np.ndarray:
        return np.isclose(sig, np.median(sig))

    # Create decorated numpy function
    @numpy_func
    def numpy_is_close_med_decorated(sig: np.ndarray) -> np.ndarray:
        return np.isclose(sig, np.median(sig))

    inp = dummy_data["TMP"]

    # Undecorated numpy function
    numpy_f = numpy_is_close_med
    assert isinstance(inp.values, np.ndarray)
    res = numpy_f(inp.values)
    assert isinstance(res, np.ndarray)
    assert res.shape == dummy_data["TMP"].shape
    assert res.dtype == np.bool8
    assert sum(res) > 0  # Check if at least 1 value is True

    # # Decorated series function
    decorated_numpy_f = numpy_is_close_med_decorated
    series_dict = series_to_series_dict(inp)
    assert isinstance(inp, pd.Series)
    res = decorated_numpy_f(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.bool8)
    assert sum(res["TMP"]) > 0  # Check if at least 1 value is True


def test_series_numpy_func_decorator(dummy_data):
    # Create undecorated series numpy function
    def normalized_freq_scale(series: pd.Series) -> np.ndarray:
        # NOTE: this is a really useless function, but it highlights a legit use case
        sr = 1 / pd.to_timedelta(pd.infer_freq(series.index)).total_seconds()
        return np.interp(series, (series.min(), series.max()), (0, sr))

    # Create decorated series numpy function
    @series_numpy_func
    def normalized_freq_scale_decorated(series: pd.Series) -> np.ndarray:
        # NOTE: this is a really useless function, but it highlights a legit use case
        sr = 1 / pd.to_timedelta(pd.infer_freq(series.index)).total_seconds()
        return np.interp(series, (series.min(), series.max()), (0, sr))

    inp = dummy_data["TMP"]

    # Undecorated numpy function
    series_numpy_f = normalized_freq_scale
    assert isinstance(inp, pd.Series)
    res = series_numpy_f(inp)
    assert isinstance(res, np.ndarray)
    assert res.shape == dummy_data["TMP"].shape
    assert res.dtype == np.float64
    assert (min(res) == 0) & (max(res) > 0)

    # # Decorated series function
    decorated_series_numpy_f = normalized_freq_scale_decorated
    series_dict = series_to_series_dict(inp)
    assert isinstance(inp, pd.Series)
    res = decorated_series_numpy_f(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.number)
    assert (min(res["TMP"]) == 0) & (max(res["TMP"]) > 0)


## SeriesProcessor


def test_single_signal_series_processor(dummy_data):
    @single_series_func
    def to_binary(series, thresh_value):
        return series.map(lambda eda: eda > thresh_value)

    thresh = 0.6
    series_processor = SeriesProcessor(["EDA"], func=to_binary, thresh_value=thresh)
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert isinstance(res, dict)
    assert res.keys() == set(["EDA"])
    assert isinstance(res["EDA"], pd.Series)
    assert res["EDA"].shape == dummy_data["EDA"].shape

    assert all(res["EDA"][dummy_data["EDA"] <= thresh] == False)
    assert all(res["EDA"][dummy_data["EDA"] > thresh] == True)


def test_multi_signal_series_processor(dummy_data):
    @single_series_func
    def percentile_clip(series, l_perc=0.01, h_perc=0.99):
        # Note: this func is useless in ML (data leakage; percentiles are not fitted)
        l_thresh = np.percentile(series, l_perc * 100)
        h_thresh = np.percentile(series, h_perc * 100)
        return series.clip(l_thresh, h_thresh)

    lower = 0.02
    upper = 0.99  # The default value => do not pass
    series_processor = SeriesProcessor(
        ["EDA", "TMP"], func=percentile_clip, l_perc=lower
    )
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert isinstance(res, dict)
    assert res.keys() == set(dummy_data.columns)
    assert isinstance(res["EDA"], pd.Series)
    assert isinstance(res["TMP"], pd.Series)
    assert res["EDA"].shape == dummy_data["EDA"].shape
    assert res["TMP"].shape == dummy_data["TMP"].shape

    assert min(res["EDA"]) == dummy_data["EDA"].quantile(lower)
    assert max(res["EDA"]) == dummy_data["EDA"].quantile(upper)
    assert min(res["TMP"]) == dummy_data["TMP"].quantile(lower)
    assert max(res["TMP"]) == dummy_data["TMP"].quantile(upper)


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
    assert any(inp["TMP"].isna())  # Check that there are some NANs present
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

    # Check if length is smaller because NANs were removed
    assert len(res_df_req) < len(dummy_data)  # Because only required signals returned
    assert len(res_dict_all["TMP"]) < len(dummy_data)  # Because no df
    assert len(res_dict_req["TMP"]) < len(dummy_data)  # Because no df
    # When merging all signals to df, the length should be the original length
    assert len(res_df_all) == len(dummy_data)

    # Check that there are no NANs present
    assert not any(res_df_req["TMP"].isna())
    assert (~any(res_dict_all["TMP"].isna())) & (~any(res_dict_req["TMP"].isna()))
    # NaNs get introduced when merging all signalsto df
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
    lower = 0.02
    upper = 0.99  # The default value => do not pass
    processing_pipeline = SeriesProcessorPipeline(
        [
            SeriesProcessor(["TMP"], func=drop_nans),
            SeriesProcessor(["TMP", "EDA"], func=percentile_clip, l_perc=lower),
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

    # Check if length is smaller because NANs were removed
    assert len(res_dict_all["TMP"]) < len(dummy_data)  # Because no df
    assert len(res_dict_req["TMP"]) < len(dummy_data)  # Because no df
    # When merging to df, the length should be the original length
    assert (len(res_df_all) == len(dummy_data)) & (len(res_df_req) == len(dummy_data))

    # Check if there are no NANs present (only valid if return_df=False)
    assert (~any(res_dict_all["TMP"].isna())) & (~any(res_dict_req["TMP"].isna()))
    # NaNs get introduced when merging to df
    assert any(res_df_all["TMP"].isna()) & any(res_df_req["TMP"].isna())

    assert all(res_dict_req["TMP"] == res_dict_all["TMP"])  # Check dict_req == dict_all
    # Check the rest against all dict_all
    # Drop NANs for df because they got introduced when merging to df
    assert all(res_df_all["TMP"].dropna().values == res_dict_all["TMP"])
    assert all(res_df_req["TMP"].dropna().values == res_dict_all["TMP"])
    assert all(res_dict_req["EDA"] == res_dict_all["EDA"])  # Check dict_req == dict_all
    # Check the rest against all dict_all
    assert all(res_df_all["EDA"] == res_dict_all["EDA"])
    assert all(res_df_req["EDA"] == res_dict_all["EDA"])

    assert min(res_dict_all["EDA"]) == dummy_data["EDA"].quantile(lower)
    assert max(res_dict_all["EDA"]) == dummy_data["EDA"].quantile(upper)
    assert min(res_dict_all["TMP"]) == inp["TMP"].quantile(lower)
    assert max(res_dict_all["TMP"]) == inp["TMP"].quantile(upper)
