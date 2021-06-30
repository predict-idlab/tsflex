"""Tests for the feature extraction functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import os
import pytest
import warnings
import pandas as pd
import numpy as np

from tsflex.features import NumpyFuncWrapper
from tsflex.features import FeatureDescriptor, MultipleFeatureDescriptors
from tsflex.features import FeatureCollection
from tsflex.features import get_feature_logs

from .utils import dummy_data, logging_file_path


test_path = os.path.abspath(os.path.dirname( __file__ ))


def test_simple_features_logging(dummy_data, logging_file_path):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(MultipleFeatureDescriptors(np.min, series_names=["TMP","ACC_x"], windows=5, strides=2.5))
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window=5, stride=2.5))

    assert set(fc.get_required_series()) == set(["EDA", "TMP", "ACC_x"])
    assert len(fc.get_required_series()) == 3

    assert not os.path.exists(logging_file_path)

    # Sequential (n_jobs <= 1), otherwise file_path gets cleared
    _ = fc.calculate(dummy_data, logging_file_path=logging_file_path, n_jobs=1)

    assert os.path.exists(logging_file_path)
    logging_df = get_feature_logs(logging_file_path)

    assert all(logging_df.columns.values == ['log_time', 'function', 'series_names', 'window', 'stride', 'duration'])

    assert len(logging_df) == 4
    assert logging_df.select_dtypes(include=[np.datetime64]).columns.values == ['log_time']
    assert logging_df.select_dtypes(include=[np.timedelta64]).columns.values == ['duration']

    assert set(logging_df["function"].values) == set(['amin', 'sum'])
    assert set(logging_df["series_names"].values) == set(["(EDA,)", "(ACC_x,)", "(TMP,)"])
    assert all(logging_df["window"] == "5s")
    assert all(logging_df["stride"] == "2.5s")


def test_file_warning_features_logging(dummy_data, logging_file_path):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )
    fc = FeatureCollection(feature_descriptors=fd)

    assert not os.path.exists(logging_file_path)

    with warnings.catch_warnings(record=True) as w:
        with open(logging_file_path, 'w'):
            pass
        assert os.path.exists(logging_file_path)
        # Sequential (n_jobs <= 1), otherwise file_path gets cleared
        _ = fc.calculate(dummy_data, logging_file_path=logging_file_path, n_jobs=1)
        assert os.path.exists(logging_file_path)
        assert len(w) == 1
        assert all([issubclass(warn.category, RuntimeWarning) for warn in w])
        assert "already exists" in str(w[0])
        # CLEANUP
        os.remove(logging_file_path)
