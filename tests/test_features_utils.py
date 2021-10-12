"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pytest
import pandas as pd
import numpy as np

from .utils import dummy_data
from tsflex.features import (
    MultipleFeatureDescriptors,
    FeatureCollection,
)
from tsflex.features.utils import make_robust


## ROBUST FUNCTIONS


def test_unrobust_gap_features(dummy_data):
    feats = MultipleFeatureDescriptors(
        functions=[np.min, np.max],
        series_names="EDA",
        windows="10s",
        strides="5s",
    )
    feature_collection = FeatureCollection(feats)

    eda_data = dummy_data["EDA"].dropna()
    eda_data[2 : 1 + 25 * 4] = None  # Leave gap of 25 s
    # -> so there are empty windows -> will rase a ValueError
    eda_data = eda_data.dropna()
    assert eda_data.isna().any() == False
    assert (eda_data.index[1:] - eda_data.index[:-1]).max() == pd.Timedelta("25 s")

    with pytest.raises(Exception):
        feature_collection.calculate(eda_data, approve_sparsity=True, n_jobs=0)


def test_robust_gap_features(dummy_data):
    feats = MultipleFeatureDescriptors(
        # here, the 'make_robust' function is used
        functions=[make_robust(f, min_nb_samples=2) for f in [np.min, np.max]],
        series_names="EDA",
        windows="10s",
        strides="5s",
    )
    feature_collection = FeatureCollection(feats)

    eda_data = dummy_data["EDA"].dropna()
    eda_data[2 : 1 + 25 * 4] = None  # Leave gap of 25 s
    eda_data = eda_data.dropna()
    # -> so there are empty windows -> will rase a ValueError
    assert eda_data.isna().any() == False
    assert (eda_data.index[1:] - eda_data.index[:-1]).max() == pd.Timedelta("25 s")

    res_df = feature_collection.calculate(
        eda_data, return_df=True, approve_sparsity=True
    )
    assert res_df.isna()[1:4].all().all()
    assert res_df[4:].isna().any().any() == False


def test_robust_gap_features_multi_input(dummy_data):
    def abs_diff_mean(x: np.ndarray, y: np.ndarray):
        abs_diff = np.abs(x - y)
        return np.mean(abs_diff)

    feats = MultipleFeatureDescriptors(
        functions=make_robust(abs_diff_mean, min_nb_samples=2),
        series_names=("EDA", "TMP"),
        windows="10s",
        strides="5s",
    )
    feature_collection = FeatureCollection(feats)

    eda_tmp_data = dummy_data[["EDA", "TMP"]].dropna()
    eda_tmp_data[2 : 1 + 25 * 4] = None  # Leave gap of 25 s
    eda_tmp_data = eda_tmp_data.dropna()
    assert eda_tmp_data.isna().any().any() == False
    assert (eda_tmp_data.index[1:] - eda_tmp_data.index[:-1]).max() == pd.Timedelta(
        "25 s"
    )

    res_df = feature_collection.calculate(
        eda_tmp_data, return_df=True, approve_sparsity=True
    )
    assert res_df.isna()[1:4].all().all()
    assert res_df[4:].isna().any().any() == False


def test_robust_gap_features_multi_output(dummy_data):
    def mean_std(x: np.ndarray):
        return np.mean(x), np.std(x)

    feats = MultipleFeatureDescriptors(
        functions=make_robust(mean_std, min_nb_samples=2, output_names=["mean", "std"]),
        series_names="EDA",
        windows="10s",
        strides="5s",
    )
    feature_collection = FeatureCollection(feats)

    eda_data = dummy_data["EDA"].dropna()
    eda_data[2 : 1 + 25 * 4] = None  # Leave gap of 25 s
    eda_data = eda_data.dropna()
    assert eda_data.isna().any() == False
    assert (eda_data.index[1:] - eda_data.index[:-1]).max() == pd.Timedelta("25 s")

    res_df = feature_collection.calculate(
        eda_data, return_df=True, approve_sparsity=True
    )
    assert res_df.isna()[1:4].all().all()
    assert res_df[4:].isna().any().any() == False


def test_unrobust_pass_through_features(dummy_data):
    # here we set the passtrough-nans attribute to True
    feats = MultipleFeatureDescriptors(
        functions=[make_robust(f, min_nb_samples=1, passthrough_nans=True) 
                   for f in [np.mean, np.min]],
        series_names="EDA",
        windows="10s",
        strides="5s",
    )
    feature_collection = FeatureCollection(feats)

    eda_data = dummy_data["EDA"].dropna()
    eda_data[np.random.choice(dummy_data.index[1:-1], 10, replace=False)] = np.nan

    res_df = feature_collection.calculate(eda_data, return_df=True)
    assert res_df.isna().any().any()


def test_robust_pass_through_features(dummy_data):
    # here we set the passtrough-nans attribute to false
    feats = MultipleFeatureDescriptors(
        functions=[make_robust(f, passthrough_nans=False) for f in [np.mean, np.min]],
        series_names="EDA",
        windows="10s",
        strides="5s",
    )
    feature_collection = FeatureCollection(feats)

    eda_data = dummy_data["EDA"].dropna()
    eda_data[np.random.choice(dummy_data.index[1:-1], 10, replace=False)] = np.nan

    res_df = feature_collection.calculate(eda_data, return_df=True)
    assert res_df.isna().any().any() == False