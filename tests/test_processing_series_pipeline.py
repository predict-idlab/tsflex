"""Tests for the processing functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import os
import pytest
import pandas as pd
import numpy as np
import dill

from tsflex.processing import dataframe_func
from tsflex.processing import SeriesProcessor, SeriesPipeline
from tsflex.processing.series_pipeline import _ProcessingError

from .utils import dummy_data

from pathlib import Path


## SeriesPipeline


def test_single_signal_series_pipeline(dummy_data):
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

    res_list_all = series_pipeline.process(inp, return_all_series=True, return_df=False)
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
    assert set([s.name for s in res_list_req]) == set(["TMP"])
    assert res_df_req.columns == ["TMP"]

    tmp_idx_all = [
        i for i in range(len(res_list_all)) if res_list_all[i].name == "TMP"
    ][0]
    tmp_idx_req = [
        i for i in range(len(res_list_req)) if res_list_req[i].name == "TMP"
    ][0]
    eda_idx_all = [
        i for i in range(len(res_list_all)) if res_list_all[i].name == "EDA"
    ][0]

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


def test_multi_signal_series_pipeline(dummy_data):
    def drop_nans(series: pd.Series) -> pd.Series:
        return series.dropna()

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
    res_list_all = series_pipeline.process(inp, return_all_series=True, return_df=False)
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
    assert set([s.name for s in res_list_req]) == set(["TMP", "EDA"])
    assert set(res_df_req.columns) == set(["TMP", "EDA"])

    # Check if length is smaller because NANs were removed
    assert all(
        [len(s) < len(dummy_data) for s in res_list_all if s.name == "TMP"]
    )  # Because no df
    assert all(
        [len(s) < len(dummy_data) for s in res_list_all if s.name == "TMP"]
    )  # Because no df
    # When merging to df, the length should be the original length
    assert (len(res_df_all) == len(dummy_data)) & (len(res_df_req) == len(dummy_data))

    tmp_idx_all = [
        i for i in range(len(res_list_all)) if res_list_all[i].name == "TMP"
    ][0]
    tmp_idx_req = [
        i for i in range(len(res_list_req)) if res_list_req[i].name == "TMP"
    ][0]
    eda_idx_all = [
        i for i in range(len(res_list_all)) if res_list_all[i].name == "EDA"
    ][0]
    eda_idx_req = [
        i for i in range(len(res_list_req)) if res_list_req[i].name == "EDA"
    ][0]

    # Check if there are no NANs present (only valid if return_df=False)
    assert ~any(res_list_all[tmp_idx_all].isna())
    assert ~any(res_list_req[tmp_idx_req].isna())
    # NaNs get introduced when merging to df
    assert any(res_df_all["TMP"].isna()) & any(res_df_req["TMP"].isna())

    assert all(
        res_list_req[tmp_idx_req] == res_list_all[tmp_idx_all]
    )  # Check list_req == req_all
    # Check the rest against all dict_all
    # Drop NANs for df because they got introduced when merging to df
    assert all(res_df_all["TMP"].dropna().values == res_list_all[tmp_idx_all])
    assert all(res_df_req["TMP"].dropna().values == res_list_all[tmp_idx_all])
    assert all(
        res_list_req[eda_idx_req] == res_list_all[eda_idx_all]
    )  # Check list_req == list_all
    # Check the rest against all dict_all
    assert all(res_df_all["EDA"] == res_list_all[eda_idx_all])
    assert all(res_df_req["EDA"] == res_list_all[eda_idx_all])

    assert min(res_list_all[eda_idx_all]) == dummy_data["EDA"].quantile(lower)
    assert max(res_list_all[eda_idx_all]) == dummy_data["EDA"].quantile(upper)
    assert min(res_list_all[tmp_idx_all]) == inp["TMP"].quantile(lower)
    assert max(res_list_all[tmp_idx_all]) == inp["TMP"].quantile(upper)


def test_pipeline_append_pipeline(dummy_data):
    pipe_a = SeriesPipeline(
        processors=[
            SeriesProcessor(lambda x: x.rolling("5min").mean(), series_names=["TMP"])
        ]
    )
    pipe_b = SeriesPipeline(
        processors=[
            SeriesProcessor(lambda x: x.rolling("5min").sum(), series_names=["TMP"])
        ]
    )

    with pytest.raises(TypeError):
        # we do not allow appending lists, must be a single item
        pipe_b.append([pipe_a])

    pipe_b.append(pipe_a)
    pipe_b.process(dummy_data)



def test_pipeline_insert(dummy_data):
    def mean(x):
        return x.rolling("5min").mean()
    def sum(x):
        return x.rolling("5min").sum()
    
    sp_mean = SeriesProcessor(mean, series_names=["TMP"])
    sp_sum = SeriesProcessor(sum, series_names=["TMP"])

    sps = [sp_sum, sp_sum, sp_mean, sp_sum, sp_sum, sp_mean, sp_mean, sp_sum, sp_mean]

    # create the base pipeline
    pipe_a = SeriesPipeline(processors=[sps[0]])
    pipe_a.insert(0, sps[1:2][0])  # note, this is a single item
    with pytest.raises(TypeError):
        # we do not allow inserting lists, must be a single item
        pipe_a.insert(0, sps[2:4])

    pipe_b = SeriesPipeline(processors=sps[2:])
    pipe_a.insert(1, pipe_b)
    p_steps = pipe_a.processing_steps
    assert p_steps[0].name == sps[0].name
    for i, ps in enumerate(p_steps[1:-1]):
        assert ps.name == sps[i+2].name
    assert p_steps[-1].name == sps[1].name

def test_pipeline_steps_operations_series_pipeline(dummy_data):
    def drop_nans(series: pd.Series) -> pd.Series:
        return series.dropna()

    def percentile_clip(series, l_perc=0.01, h_perc=0.99):
        # Note: this func is useless in ML (data leakage; percentiles are not fitted)
        l_thresh = np.percentile(series, l_perc * 100)
        h_thresh = np.percentile(series, h_perc * 100)
        return series.clip(l_thresh, h_thresh)

    lower = 0.02
    series_pipeline = SeriesPipeline()

    assert series_pipeline.get_required_series() == []
    assert series_pipeline.processing_steps == []

    series_pipeline.append(
        SeriesProcessor(
            series_names=["TMP", "EDA"],
            function=percentile_clip,
            l_perc=lower,
        )
    )

    assert len(series_pipeline.get_required_series()) == 2
    assert set(series_pipeline.get_required_series()) == set(["TMP", "EDA"])
    assert len(series_pipeline.processing_steps) == 1
    assert series_pipeline.processing_steps[0].name == "percentile_clip"

    series_pipeline.insert(
        idx=0, processor=SeriesProcessor(series_names=["ACC_x"], function=drop_nans)
    )

    assert len(series_pipeline.get_required_series()) == 3
    assert set(series_pipeline.get_required_series()) == set(["TMP", "EDA", "ACC_x"])
    assert len(series_pipeline.processing_steps) == 2
    assert series_pipeline.processing_steps[0].name == "drop_nans"
    assert series_pipeline.processing_steps[1].name == "percentile_clip"


def test_pipeline_drop_keys_series_pipeline(dummy_data):
    def percentile_clip(series, l_perc=0.01, h_perc=0.99):
        # Note: this func is useless in ML (data leakage; percentiles are not fitted)
        l_thresh = np.percentile(series, l_perc * 100)
        h_thresh = np.percentile(series, h_perc * 100)
        return series.clip(l_thresh, h_thresh)

    series_pipeline = SeriesPipeline()

    assert series_pipeline.get_required_series() == []
    assert series_pipeline.processing_steps == []

    lower = 0.02
    series_pipeline.append(
        SeriesProcessor(
            series_names=["TMP", "EDA", "ACC_x"],
            function=percentile_clip,
            l_perc=lower,
        )
    )

    assert len(series_pipeline.get_required_series()) == 3
    assert set(series_pipeline.get_required_series()) == set(["TMP", "EDA", "ACC_x"])
    assert len(series_pipeline.processing_steps) == 1
    assert series_pipeline.processing_steps[0].name == "percentile_clip"

    res = series_pipeline.process(
        dummy_data, return_all_series=True, drop_keys=["ACC_z"], return_df=True
    )
    assert len(res.columns) == 4
    assert set(res.columns) == set(["TMP", "EDA", "ACC_x", "ACC_y"])


def test_pipeline_processors_are_pipelines(dummy_data):
    def percentile_clip(series, l_perc=0.01, h_perc=0.99):
        # Note: this func is useless in ML (data leakage; percentiles are not fitted)
        l_thresh = np.percentile(series, l_perc * 100)
        h_thresh = np.percentile(series, h_perc * 100)
        return series.clip(l_thresh, h_thresh)

    lower = 0.02
    series_pipeline = SeriesPipeline(
        processors=[
            SeriesPipeline(
                processors=[
                    SeriesProcessor(
                        series_names=["TMP", "EDA", "ACC_x"],
                        function=percentile_clip,
                        l_perc=lower,
                    )
                ]
            )
        ]
    )

    assert len(series_pipeline.get_required_series()) == 3
    assert set(series_pipeline.get_required_series()) == set(["TMP", "EDA", "ACC_x"])
    assert len(series_pipeline.processing_steps) == 1
    assert series_pipeline.processing_steps[0].name == "percentile_clip"

    res = series_pipeline.process(
        dummy_data, return_all_series=True, drop_keys=["ACC_z"], return_df=True
    )
    assert len(res.columns) == 4
    assert set(res.columns) == set(["TMP", "EDA", "ACC_x", "ACC_y"])

    # let's also test the repr string
    print(series_pipeline)


### Test error use-case


def test_error_input_dataframe_no_time_index_series_pipeline(dummy_data):
    def absx2(sig):
        return np.abs(sig * 2)

    # Check if abs_diff returns a series without a name
    absx2_tmp = absx2(dummy_data["TMP"])
    assert isinstance(absx2_tmp, pd.Series)
    assert absx2_tmp.name == "TMP"
    assert isinstance(absx2_tmp.index, pd.DatetimeIndex)

    series_pipeline = SeriesPipeline(
        [
            SeriesProcessor(
                series_names=["EDA", "TMP"],
                function=absx2,
            )
        ]
    )

    inp = dummy_data.reset_index()
    assert not isinstance(inp.index, pd.DatetimeIndex)

    with pytest.raises(Exception):
        _ = series_pipeline.process(inp)


# Test throwing of _ProcessingError by using basic case
def test_error_output_dataframe_no_time_index_series_pipeline(dummy_data):
    def absx2(sig):
        return pd.DataFrame(np.abs(sig.values * 2), columns=[sig.name])

    absx2_tmp = absx2(dummy_data["TMP"])
    assert isinstance(absx2_tmp, pd.DataFrame)
    assert absx2_tmp.columns.values == ["TMP"]
    assert not isinstance(absx2_tmp.index, pd.DatetimeIndex)

    series_pipeline = SeriesPipeline(
        [
            SeriesProcessor(
                series_names=["EDA", "TMP"],
                function=absx2,
            )
        ]
    )

    with pytest.raises(_ProcessingError):
        _ = series_pipeline.process(dummy_data)


def test_serialization(dummy_data):
    def drop_nans(series: pd.Series) -> pd.Series:
        return series.dropna()

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
                h_perc=upper,
            ),
        ]
    )

    res_df_all = series_pipeline.process(inp, return_all_series=False, return_df=True)
    col_order = res_df_all.columns

    save_path = Path("series_pipeline.pkl")
    if save_path.exists():
        os.remove(save_path)
    assert not save_path.exists()
    series_pipeline.serialize(save_path)
    assert save_path.exists() and save_path.is_file()

    sp_deserialized: SeriesPipeline = dill.load(open(save_path, "rb"))
    out_deserialized = sp_deserialized.process(
        inp, return_all_series=False, return_df=True
    )
    assert np.allclose(
        res_df_all[col_order].values.astype(float),
        out_deserialized[col_order].values.astype(float),
        equal_nan=True,
    )
    os.remove(save_path)
