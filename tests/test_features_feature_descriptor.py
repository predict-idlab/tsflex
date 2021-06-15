"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pytest
import pandas as pd
import numpy as np

from tsflex.features import NumpyFuncWrapper
from tsflex.features import FeatureDescriptor, MultipleFeatureDescriptors

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
    assert isinstance(fd.function, NumpyFuncWrapper)

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
    assert isinstance(fd.function, NumpyFuncWrapper)

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
    assert isinstance(fd.function, NumpyFuncWrapper)

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
    assert isinstance(fd.function, NumpyFuncWrapper)

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

def test_multiple_feature_descriptors():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    mfd = MultipleFeatureDescriptors(
        functions=[sum_func, NumpyFuncWrapper(np.max), np.min],
        series_names=["EDA", "TMP"],
        windows=['5s', '7.5s'],
        strides='2.5s',
    )

    assert len(mfd.feature_descriptions) == 3*2*2

    series_names = [fd.series_name for fd in mfd.feature_descriptions]
    assert set(series_names) == set([tuple(["EDA"]), tuple(["TMP"])])
    assert sum([el == tuple(["EDA"]) for el in series_names]) == 3*2
    assert sum([el == tuple(["TMP"]) for el in series_names]) == 3*2

    windows = [fd.window for fd in mfd.feature_descriptions]
    assert set(windows) == set([pd.Timedelta(seconds=5), pd.Timedelta(seconds=7.5)])
    assert sum([el == pd.Timedelta(seconds=5) for el in windows]) == 3*2
    assert sum([el == pd.Timedelta(seconds=7.5) for el in windows]) == 3*2

    strides = [fd.stride for fd in mfd.feature_descriptions]
    assert (set(strides) == set([pd.Timedelta(seconds=2.5)]))

    functions = [fd.function for fd in mfd.feature_descriptions]
    assert len(set(functions)) == 3
    output_names = [f.output_names for f in functions]
    assert all([len(outputs) == 1 for outputs in output_names])
    output_names = [outputs[0] for outputs in output_names]
    assert set(output_names) == set(['sum_func', 'amax', 'amin'])
    assert sum([el == 'sum_func' for el in output_names]) == 2*2
    assert sum([el == 'amax' for el in output_names]) == 2*2
    assert sum([el == 'amin' for el in output_names]) == 2*2
