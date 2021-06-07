"""Tests for the processing functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd
import numpy as np

from time_series.processing import dataframe_func
from time_series.processing import SeriesProcessor, SeriesPipeline

from .utils import dummy_data, dataframe_to_series_dict, series_to_series_dict


## Function wrappers


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
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (dummy_data.shape[0] - 10,) + dummy_data.shape[1:]
    assert not np.any(pd.isna(res))  # Check that there are no NANs present


## Various output types for single_series_func = True


def test_dataframe_output(dummy_data):
    # Create dataframe output function
    def duplicate_with_offset(series_dict, offset: float) -> pd.DataFrame:
        offset = abs(offset)
        df = pd.DataFrame()
        df["TMP" + "+" + str(offset)] = series_dict["TMP"] + offset
        df["TMP" + "-" + str(offset)] = series_dict["TMP"] - offset
        return df

    dataframe_f = duplicate_with_offset
    offset = 0.5

    inp = dummy_data["TMP"]
    assert isinstance(inp, pd.Series)
    series_dict = series_to_series_dict(inp)
    assert isinstance(series_dict, dict)

    # Raw series function
    res = dataframe_f(series_dict, offset)
    assert isinstance(res, pd.DataFrame)
    assert (res.shape[0] == len(dummy_data["TMP"])) & (res.shape[1] == 2)
    assert np.all(res[f"TMP+{offset}"] == inp + offset)
    assert np.all(res[f"TMP-{offset}"] == inp - offset)

    # SeriesProcessor with series function
    processor = SeriesProcessor(["TMP"], func=dataframe_f, offset=offset)
    res = processor(series_dict)
    assert isinstance(res, pd.DataFrame)
    assert (res.shape[0] == len(dummy_data["TMP"])) & (res.shape[1] == 2)
    assert np.all(res[f"TMP+{offset}"] == inp + offset)
    assert np.all(res[f"TMP-{offset}"] == inp - offset)


def test_series_output(dummy_data):
    # Create series output function
    def to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series)

    single_series_f = to_numeric

    inp = dummy_data["TMP"].astype(str)
    assert isinstance(inp, pd.Series)

    # Undecorated series function
    assert not np.issubdtype(inp, np.number)
    res = single_series_f(inp)
    assert isinstance(res, pd.Series)
    assert res.shape == dummy_data["TMP"].shape
    assert np.issubdtype(res, np.number)

    # Decorated series function
    processor = SeriesProcessor(["TMP"], func=single_series_f, single_series_func=True)
    series_dict = series_to_series_dict(inp)
    assert not np.issubdtype(series_dict["TMP"], np.number)
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.number)


def test_numpy_func(dummy_data):
    # Create numpy function
    def numpy_is_close_med(sig: np.ndarray) -> np.ndarray:
        return np.isclose(sig, np.median(sig))

    numpy_f = numpy_is_close_med

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
    processor = SeriesProcessor(["TMP"], func=numpy_f, single_series_func=True)
    series_dict = series_to_series_dict(inp)
    assert isinstance(inp, pd.Series)
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.bool8)
    assert sum(res["TMP"]) > 0  # Check if at least 1 value is True


def test_series_numpy_func(dummy_data):
    # Create series numpy function
    def normalized_freq_scale(series: pd.Series) -> np.ndarray:
        # NOTE: this is a really useless function, but it highlights a legit use case
        sr = 1 / pd.to_timedelta(pd.infer_freq(series.index)).total_seconds()
        return np.interp(series, (series.min(), series.max()), (0, sr))

    series_numpy_f = normalized_freq_scale

    inp = dummy_data["TMP"]

    # Undecorated numpy function
    assert isinstance(inp, pd.Series)
    res = series_numpy_f(inp)
    assert isinstance(res, np.ndarray)
    assert res.shape == dummy_data["TMP"].shape
    assert res.dtype == np.float64
    assert (min(res) == 0) & (max(res) > 0)

    # # Decorated series function
    processor = SeriesProcessor(["TMP"], func=series_numpy_f, single_series_func=True)
    series_dict = series_to_series_dict(inp)
    assert isinstance(inp, pd.Series)
    res = processor(series_dict)
    assert isinstance(res, dict)
    assert res.keys() == series_dict.keys()
    assert isinstance(res["TMP"], pd.Series)
    assert res["TMP"].shape == dummy_data["TMP"].shape
    assert np.issubdtype(res["TMP"], np.number)
    assert (min(res["TMP"]) == 0) & (max(res["TMP"]) > 0)


## SeriesProcessor


def test_single_signal_series_processor(dummy_data):
    def to_binary(series, thresh_value):
        return series.map(lambda eda: eda > thresh_value)

    thresh = 0.6
    series_processor = SeriesProcessor(
        ["EDA"], func=to_binary, single_series_func=True, thresh_value=thresh
    )
    series_dict = dataframe_to_series_dict(dummy_data)
    res = series_processor(series_dict)

    assert isinstance(res, dict)
    assert res.keys() == set(["EDA"])
    assert isinstance(res["EDA"], pd.Series)
    assert res["EDA"].shape == dummy_data["EDA"].shape

    assert all(res["EDA"][dummy_data["EDA"] <= thresh] == False)
    assert all(res["EDA"][dummy_data["EDA"] > thresh] == True)


def test_multi_signal_series_processor(dummy_data):
    def percentile_clip(series, l_perc=0.01, h_perc=0.99):
        # Note: this func is useless in ML (data leakage; percentiles are not fitted)
        l_thresh = np.percentile(series, l_perc * 100)
        h_thresh = np.percentile(series, h_perc * 100)
        return series.clip(l_thresh, h_thresh)

    lower = 0.02
    upper = 0.99  # The default value => do not pass
    series_processor = SeriesProcessor(
        ["EDA", "TMP"], func=percentile_clip, l_perc=lower, single_series_func=True
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


## SeriesPipeline


def test_single_signal_series_processor_pipeline(dummy_data):
    def interpolate(series: pd.Series) -> pd.Series:
        return series.interpolate()

    # TODO: dit als dataframe_func houden?
    @dataframe_func  # Note: this can (and should) be a single_series_func as dropna works on pd.Series
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    inp = dummy_data.copy()
    inp.loc[inp["TMP"] > 31.5, "TMP"] = pd.NA
    assert any(inp["TMP"].isna())  # Check that there are some NANs present
    series_pipeline = SeriesPipeline(
        [
            SeriesProcessor(["TMP"], func=interpolate, single_series_func=True),
            SeriesProcessor(["TMP"], func=drop_nans),
        ]
    )
    res_list_all = series_pipeline.process(
        inp, return_all_series=True, return_df=False
    )
    res_list_req = series_pipeline.process(
        inp, return_all_series=False, return_df=False
    )
    res_df_all = series_pipeline.process(inp, return_all_series=True, return_df=True)
    res_df_req = series_pipeline.process(inp, return_all_series=False, return_df=True)

    assert isinstance(res_list_all, list) & isinstance(res_list_req, list)
    assert isinstance(res_df_all, pd.DataFrame) & isinstance(res_df_req, pd.DataFrame)

    assert set([s.name for s in res_list_all]) == set(["TMP", "EDA"])
    assert set(res_df_all.columns) == set(["TMP", "EDA"])
    assert set([s.name for s in res_list_req])== set(["TMP"])
    assert res_df_req.columns == ["TMP"]

    tmp_idx_all = [i for i in range(len(res_list_all)) if res_list_all[i].name == "TMP"][0]
    tmp_idx_req = [i for i in range(len(res_list_req)) if res_list_req[i].name == "TMP"][0]
    eda_idx_all = [i for i in range(len(res_list_all)) if res_list_all[i].name == "EDA"][0]

    # Check if length is smaller because NANs were removed
    assert len(res_df_req) < len(dummy_data)  # Because only required signals returned
    assert len(res_list_all[tmp_idx_all]) < len(dummy_data)  # Because no df
    assert len(res_list_req[tmp_idx_req]) < len(dummy_data)  # Because no df
    # When merging all signals to df, the length should be the original length
    assert len(res_df_all) == len(dummy_data)

    # Check that there are no NANs present
    assert not any(res_df_req["TMP"].isna())
    assert ~any(res_list_all[tmp_idx_all].isna())
    assert ~any(res_list_req[tmp_idx_req].isna())
    # NaNs get introduced when merging all signalsto df
    assert any(res_df_all["TMP"].isna())

    assert all(res_df_all["TMP"].dropna().values == res_list_all[tmp_idx_all])
    assert all(res_df_req["TMP"].values == res_list_all[tmp_idx_all])

    assert all(res_list_all[eda_idx_all] == inp["EDA"])
    assert all(res_df_all["EDA"] == inp["EDA"])


def test_multi_signal_series_processor_pipeline(dummy_data):
    # TODO: dit als dataframe_func houden?
    @dataframe_func  # Note: this can (and should) be a single_series_func as dropna works on pd.Series
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

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
    series_pipeline = SeriesPipeline(
        [
            SeriesProcessor(["TMP"], func=drop_nans),
            SeriesProcessor(
                ["TMP", "EDA"],
                func=percentile_clip,
                single_series_func=True,
                l_perc=lower,
            ),
        ]
    )
    res_list_all = series_pipeline.process(
        inp, return_all_series=True, return_df=False
    )
    res_list_req = series_pipeline.process(
        inp, return_all_series=False, return_df=False
    )
    res_df_all = series_pipeline.process(inp, return_all_series=True, return_df=True)
    res_df_req = series_pipeline.process(inp, return_all_series=False, return_df=True)

    assert isinstance(res_list_all, list) & isinstance(res_list_req, list)
    assert isinstance(res_df_all, pd.DataFrame) & isinstance(res_df_req, pd.DataFrame)
    assert set([s.name for s in res_list_all]) == set(["TMP", "EDA"])
    assert set(res_df_all.columns) == set(["TMP", "EDA"])
    assert set([s.name for s in res_list_req])== set(["TMP", "EDA"])
    assert set(res_df_req.columns) == set(["TMP", "EDA"])

    # Check if length is smaller because NANs were removed
    assert all([len(s) < len(dummy_data) for s in res_list_all if s.name == "TMP"])  # Because no df
    assert all([len(s) < len(dummy_data) for s in res_list_all if s.name == "TMP"])  # Because no df
    # When merging to df, the length should be the original length
    assert (len(res_df_all) == len(dummy_data)) & (len(res_df_req) == len(dummy_data))

    tmp_idx_all = [i for i in range(len(res_list_all)) if res_list_all[i].name == "TMP"][0]
    tmp_idx_req = [i for i in range(len(res_list_req)) if res_list_req[i].name == "TMP"][0]
    eda_idx_all = [i for i in range(len(res_list_all)) if res_list_all[i].name == "EDA"][0]
    eda_idx_req = [i for i in range(len(res_list_req)) if res_list_req[i].name == "EDA"][0]

    # Check if there are no NANs present (only valid if return_df=False)
    assert ~any(res_list_all[tmp_idx_all].isna()) 
    assert ~any(res_list_req[tmp_idx_req].isna())
    # NaNs get introduced when merging to df
    assert any(res_df_all["TMP"].isna()) & any(res_df_req["TMP"].isna())

    assert all(res_list_req[tmp_idx_req] == res_list_all[tmp_idx_all])  # Check list_req == req_all
    # Check the rest against all dict_all
    # Drop NANs for df because they got introduced when merging to df
    assert all(res_df_all["TMP"].dropna().values == res_list_all[tmp_idx_all])
    assert all(res_df_req["TMP"].dropna().values == res_list_all[tmp_idx_all])
    assert all(res_list_req[eda_idx_req] == res_list_all[eda_idx_all])  # Check list_req == list_all
    # Check the rest against all dict_all
    assert all(res_df_all["EDA"] == res_list_all[eda_idx_all])
    assert all(res_df_req["EDA"] == res_list_all[eda_idx_all])

    assert min(res_list_all[eda_idx_all]) == dummy_data["EDA"].quantile(lower)
    assert max(res_list_all[eda_idx_all]) == dummy_data["EDA"].quantile(upper)
    assert min(res_list_all[tmp_idx_all]) == inp["TMP"].quantile(lower)
    assert max(res_list_all[tmp_idx_all]) == inp["TMP"].quantile(upper)


# TODO: test drop_keys
