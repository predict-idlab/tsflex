"""Tests for the processing functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd
import numpy as np

from tsflex.processing import dataframe_func
from tsflex.processing import SeriesProcessor, SeriesPipeline

from .utils import dummy_data, dataframe_to_series_dict, series_to_series_dict


## SeriesPipeline


def test_single_signal_series_processor_pipeline(dummy_data):
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
            SeriesProcessor(series_names=["TMP"], function=interpolate),
            SeriesProcessor(series_names=["TMP"], function=drop_nans),
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

    assert series_pipeline.get_required_series() == ["TMP"]

    assert isinstance(res_list_all, list) & isinstance(res_list_req, list)
    assert isinstance(res_df_all, pd.DataFrame) & isinstance(res_df_req, pd.DataFrame)

    assert set([s.name for s in res_list_all]) == set(dummy_data.columns)
    assert set(res_df_all.columns) == set(dummy_data.columns)
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
    # NaNs get introduced when merging all signals to df
    assert any(res_df_all["TMP"].isna())

    assert all(res_df_all["TMP"].dropna().values == res_list_all[tmp_idx_all])
    assert all(res_df_req["TMP"].values == res_list_all[tmp_idx_all])

    assert all(res_list_all[eda_idx_all] == inp["EDA"])
    assert all(res_df_all["EDA"] == inp["EDA"])


def test_multi_signal_series_processor_pipeline(dummy_data):
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
            SeriesProcessor(series_names=["TMP"], function=drop_nans),
            SeriesProcessor(
                series_names=["TMP", "EDA"],
                function=percentile_clip,
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

    assert len(series_pipeline.get_required_series()) == 2
    assert set(series_pipeline.get_required_series()) == set(["TMP", "EDA"])

    assert isinstance(res_list_all, list) & isinstance(res_list_req, list)
    assert isinstance(res_df_all, pd.DataFrame) & isinstance(res_df_req, pd.DataFrame)
    assert set([s.name for s in res_list_all]) == set(dummy_data.columns)
    assert set(res_df_all.columns) == set(dummy_data.columns)
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




# TODO: test append
# TODO: test insert
# TODO: test get_all_required_series
# TODO: test serialize

# TODO: test drop_keys
# TODO: Test skseries_processing_pipeline