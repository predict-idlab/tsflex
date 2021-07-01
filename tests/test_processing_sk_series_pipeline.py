"""Tests for the processing functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pytest
import pandas as pd
import numpy as np

from tsflex.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from tsflex.processing import dataframe_func
from tsflex.processing import SeriesProcessor
from tsflex.pipeline import SKSeriesPipeline

from .utils import dummy_data, pipe_transform


## SKSeriesPipeline


def test_single_signal_sk_series_pipeline_standalone(dummy_data):
    def interpolate(series: pd.Series) -> pd.Series:
        return series.interpolate()

    @dataframe_func  # Note: this can (and should) be a single_series_func as dropna works on pd.Series
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    inp = dummy_data.copy()
    inp.loc[inp["TMP"] > 31.5, "TMP"] = pd.NA
    assert any(inp["TMP"].isna())  # Check that there are some NANs present
    processors = [
        SeriesProcessor(series_names="TMP", function=interpolate),
        SeriesProcessor(series_names="TMP", function=drop_nans),
    ]
    series_pipeline1 = SKSeriesPipeline(processors, return_all_series=True)
    series_pipeline2 = SKSeriesPipeline(
        processors, return_all_series=True, drop_keys=["ACC_x"]
    )
    series_pipeline3 = SKSeriesPipeline(processors, return_all_series=False)

    res_all1 = series_pipeline1.transform(inp)
    res_all1_fit = series_pipeline1.fit_transform(inp)
    res_all2 = series_pipeline2.transform(inp)
    res_req = series_pipeline3.transform(inp)
    results = [res_all1, res_all1_fit, res_all2, res_req]

    assert all([isinstance(res, pd.DataFrame) for res in results])
    assert len(res_all1.columns) == len(dummy_data.columns)
    assert set(res_all1.columns.values) == set(dummy_data.columns.values)
    assert len(res_all1_fit.columns) == len(dummy_data.columns)
    assert set(res_all1_fit.columns.values) == set(dummy_data.columns.values)
    assert len(res_all2.columns) == len(dummy_data.columns) - 1
    assert set(res_all2.columns.values) == set(dummy_data.columns.values).difference(
        ["ACC_x"]
    )
    assert res_req.columns.values == ["TMP"]

    # Check if length is smaller because NANs were removed
    assert len(res_req) < len(dummy_data)  # Because only required signals returned
    # When merging all signals to df, the length should be the original length
    assert (
        len(set([len(res_all1), len(res_all1_fit), len(res_all2), len(dummy_data)]))
        == 1
    )

    # Check that there are no NANs present
    assert not any(res_req["TMP"].isna())
    # NaNs get introduced when merging all signals to df
    assert (
        any(res_all1["TMP"].isna())
        & any(res_all1_fit["TMP"].isna())
        & any(res_all2["TMP"].isna())
    )

    assert all(res_all1["TMP"].dropna().values == res_req["TMP"])
    assert all(res_all1_fit["TMP"].dropna().values == res_req["TMP"])
    assert all(res_all2["TMP"].dropna().values == res_req["TMP"])

    assert all(res_all1["EDA"] == inp["EDA"])
    assert all(res_all1_fit["EDA"] == inp["EDA"])
    assert all(res_all2["EDA"] == inp["EDA"])


def test_simple_learning_pipeline_start(dummy_data):
    def acc_sum(acc1, acc2, acc3) -> pd.Series:
        abs_sum = np.abs(acc1) + np.abs(acc2) + np.abs(acc3)
        return pd.Series(abs_sum, name="ACC_abs_sum")

    processors = [
        SeriesProcessor(series_names=("ACC_x", "ACC_y", "ACC_z"), function=acc_sum),
    ]
    series_pipeline = SKSeriesPipeline(
        processors, return_all_series=True, drop_keys=["ACC_x", "ACC_y", "ACC_z"]
    )

    pipeline = make_pipeline(
        steps=[
            ("processing", series_pipeline),
            ("impute", SimpleImputer(strategy="constant")),
            ("scale", StandardScaler()),
            ("lin_reg", LinearRegression())
        ]
    )

    pipeline.fit(dummy_data.drop(columns=["EDA"]), dummy_data["EDA"])
    # Lovely data leakage :)
    y_pred = pipeline.predict(dummy_data.drop(columns=["EDA"]))
    assert len(y_pred) == len(dummy_data)
    # Assert that the pipeline learned something (with data leakage ofc)
    assert pipeline.score(dummy_data.drop(columns=["EDA"]), dummy_data["EDA"]) > 0

    transformed = pipe_transform(pipeline, dummy_data.drop(columns=["EDA"]))
    assert transformed.shape == (len(dummy_data), 2)
    assert len(np.mean(transformed, axis=0)) == 2
    assert all(np.isclose(np.mean(transformed, axis=0), 0))
    assert all(np.isclose(np.std(transformed, axis=0), 1))


def test_simple_learning_pipeline_middle(dummy_data):
    def acc_sum(acc1, acc2, acc3) -> pd.Series:
        abs_sum = np.abs(acc1) + np.abs(acc2) + np.abs(acc3)
        return pd.Series(abs_sum, name="ACC_abs_sum")

    processors = [
        SeriesProcessor(series_names=("ACC_x", "ACC_y", "ACC_z"), function=acc_sum),
    ]
    series_pipeline = SKSeriesPipeline(
        processors, return_all_series=True, drop_keys=["ACC_x", "ACC_y", "ACC_z"]
    )

    pipeline = make_pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("processing", series_pipeline),
            ("scale", StandardScaler()),
            ("lin_reg", LinearRegression())
        ]
    )

    pipeline.fit(dummy_data.drop(columns=["EDA"]), dummy_data["EDA"])

    # Lovely data leakage :)
    y_pred = pipeline.predict(dummy_data.drop(columns=["EDA"]))
    assert len(y_pred) == len(dummy_data)
    # Assert that the pipeline learned something (with data leakage ofc)
    assert pipeline.score(dummy_data.drop(columns=["EDA"]), dummy_data["EDA"]) > 0

    transformed = pipe_transform(pipeline, dummy_data.drop(columns=["EDA"]))
    assert transformed.shape == (len(dummy_data), 2)
    assert len(np.mean(transformed, axis=0)) == 2
    assert all(np.isclose(np.mean(transformed, axis=0), 0))
    assert all(np.isclose(np.std(transformed, axis=0), 1))
