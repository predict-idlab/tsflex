"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd
import numpy as np

from tsflex.features import NumpyFuncWrapper
from tsflex.features import FeatureDescriptor, MultipleFeatureDescriptors
from typing import Tuple

from .utils import dummy_data, dataframe_to_series_dict, series_to_series_dict


### FeatureDescriptor

def test_simple_feature_descriptor():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
        window='5s',
        stride='2.5s',
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit='seconds')
    assert fd.stride == pd.Timedelta(2.5, unit='seconds')
    assert fd.get_required_series() == ["EDA"]

def test_simple_feature_descriptor_float_seconds():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
        window=5,
        stride=2.5,
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit='seconds')
    assert fd.stride == pd.Timedelta(2.5, unit='seconds')
    assert fd.get_required_series() == ["EDA"]
