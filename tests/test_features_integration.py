"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import seglearn
import tsfresh
import pandas as pd
import numpy as np

from .utils import dummy_data
from tsflex.features import (
    MultipleFeatureDescriptors,
    FeatureCollection,
    FuncWrapper,
    feature_collection,
)
from tsflex.features.integrations import seglearn_wrapper, tsfresh_combiner_wrapper


## SEGLEARN


def test_seglearn_basic_features(dummy_data):
    base_features = seglearn.feature_functions.base_features

    basic_feats = MultipleFeatureDescriptors(
        functions=[seglearn_wrapper(f, k) for k, f in base_features().items()],
        series_names=["TMP", "EDA"],
        windows="5min",
        strides="2min",
    )
    feature_collection = FeatureCollection(basic_feats)

    res_df = feature_collection.calculate(dummy_data, return_df=True)
    assert res_df.shape[1] == len(base_features()) * 2
    assert res_df.shape[0] > 0
    assert res_df.isna().any().any() == False


def test_seglearn_all_features(dummy_data):
    all_features = seglearn.feature_functions.all_features

    all_feats = MultipleFeatureDescriptors(
        functions=[
            seglearn_wrapper(f, k) for k, f in all_features().items() if k != "hist4"
        ]
        + [
            seglearn_wrapper(all_features()["hist4"], [f"hist{i}" for i in range(1, 5)])
        ],
        series_names=["TMP", "EDA"],
        windows="5min",
        strides="2min",
    )
    feature_collection = FeatureCollection(all_feats)

    res_df = feature_collection.calculate(dummy_data, return_df=True)
    assert res_df.shape[1] == (len(all_features()) - 1 + 4) * 2
    assert res_df.shape[0] > 0
    assert res_df.isna().any().any() == False


## TSFRESH


def test_tsfresh_simple_features(dummy_data):
    from tsfresh.feature_extraction.feature_calculators import (
        abs_energy,
        absolute_sum_of_changes,
        cid_ce,
        variance_larger_than_standard_deviation,
    )

    simple_feats = MultipleFeatureDescriptors(
        functions=[
            abs_energy,
            absolute_sum_of_changes,
            variance_larger_than_standard_deviation,
            FuncWrapper(cid_ce, normalize=True),
        ],
        series_names=["ACC_x", "EDA"],
        windows="5min",
        strides="2min",
    )
    feature_collection = FeatureCollection(simple_feats)

    res_df = feature_collection.calculate(dummy_data, return_df=True)
    assert res_df.shape[1] == 4 * 2
    assert res_df.shape[0] > 0
    assert res_df.isna().any().any() == False


def test_tsfresh_combiner_features(dummy_data):
    from tsfresh.feature_extraction.feature_calculators import (
        index_mass_quantile,
        linear_trend,
        spkt_welch_density,
        linear_trend_timewise,
    )

    combiner_feats = MultipleFeatureDescriptors(
        functions=[
            tsfresh_combiner_wrapper(
                index_mass_quantile, param=[{"q": v} for v in [0.15, 0.5, 0.75]]
            ),
            tsfresh_combiner_wrapper(
                linear_trend,
                param=[{"attr": v} for v in ["intercept", "slope", "stderr"]],
            ),
            tsfresh_combiner_wrapper(
                spkt_welch_density, param=[{"coeff": v} for v in range(5)]
            ),
            # This function requires a pd.Series with a pd.DatetimeIndex
            tsfresh_combiner_wrapper(
                linear_trend_timewise,
                param=[{"attr": v} for v in ["intercept", "slope"]],
            ),
        ],
        series_names=["ACC_x", "EDA"],
        windows="5min",
        strides="2min",
    )
    feature_collection = FeatureCollection(combiner_feats)

    res_df = feature_collection.calculate(dummy_data, return_df=True)
    assert res_df.shape[1] == (3 + 3 + 5 + 2) * 2
    assert res_df.shape[0] > 0
    assert res_df.isna().any().any() == False
