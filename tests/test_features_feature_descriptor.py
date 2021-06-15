"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd
import numpy as np
import pytest

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

def test_simple_raw_np_func_feature_descriptor():
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window='5s',
        stride='2.5s',
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit='seconds')
    assert fd.stride == pd.Timedelta(2.5, unit='seconds')
    assert fd.get_required_series() == ["EDA"]

def test_simple_feature_descriptor_str_float_seconds():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
        window='5',
        stride=2.5,
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit='seconds')
    assert fd.stride == pd.Timedelta(2.5, unit='seconds')
    assert fd.get_required_series() == ["EDA"]

def test_simple_feature_descriptor_func_wrapper():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    sum_func_wrapped = NumpyFuncWrapper(sum_func)

    fd = FeatureDescriptor(
        function=sum_func_wrapped,
        series_name="EDA",
        window='5',
        stride='2.5s',
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit='seconds')
    assert fd.stride == pd.Timedelta(2.5, unit='seconds')
    assert fd.get_required_series() == ["EDA"]

def test_error_function_simple_feature_descriptor():
    invalid_func = []  # Something that is not callable

    with pytest.raises(TypeError):
        _ = FeatureDescriptor(
            function=invalid_func,
            series_name="EDA",
            window=5,
            stride='2.5s',
        )

def test_error_time_arg_simple_feature_descriptor():
    invalid_stride = pd.to_datetime('13000101', format='%Y%m%d', errors='ignore')
    with pytest.raises(TypeError):
        _ = FeatureDescriptor(
            function=np.sum,
            series_name="EDA",
            window=5,
            stride=invalid_stride,
        )

### MultipleFeatureDescriptors

