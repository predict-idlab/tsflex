"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import numpy as np
import pandas as pd
import pytest

from tsflex.features import FeatureDescriptor, FuncWrapper, MultipleFeatureDescriptors
from tsflex.utils.data import flatten

### FeatureDescriptor


def test_simple_feature_descriptor():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit="seconds")
    assert fd.stride == [pd.Timedelta(2.5, unit="seconds")]
    assert fd.get_required_series() == ["EDA"]
    assert fd.get_nb_output_features() == 1
    assert isinstance(fd.function, FuncWrapper)


def test_simple_raw_np_func_feature_descriptor():
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit="seconds")
    assert fd.stride == [pd.Timedelta(2.5, unit="seconds")]
    assert fd.get_required_series() == ["EDA"]
    assert fd.get_nb_output_features() == 1
    assert isinstance(fd.function, FuncWrapper)


def test_simple_feature_descriptor_optional_stride():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
        window="5s",
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit="seconds")
    assert fd.stride is None
    assert fd.get_required_series() == ["EDA"]
    assert fd.get_nb_output_features() == 1
    assert isinstance(fd.function, FuncWrapper)


def test_simple_feature_descriptor_optional_window_and_stride():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window is None
    assert fd.stride is None
    assert fd.get_required_series() == ["EDA"]
    assert fd.get_nb_output_features() == 1
    assert isinstance(fd.function, FuncWrapper)


def test_simple_feature_descriptor_multiple_strides():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
        window="5s",
        stride=["3s", "5s"],
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit="seconds")
    assert fd.stride == [
        pd.Timedelta(3, unit="seconds"),
        pd.Timedelta(5, unit="seconds"),
    ]
    assert fd.get_required_series() == ["EDA"]
    assert fd.get_nb_output_features() == 1
    assert isinstance(fd.function, FuncWrapper)


def test_simple_feature_descriptor_floats():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
        window=5.0,
        stride=2.5,
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == 5.0
    assert fd.stride == [2.5]
    assert fd.get_required_series() == ["EDA"]
    assert fd.get_nb_output_features() == 1
    assert isinstance(fd.function, FuncWrapper)


def test_simple_feature_descriptor_str_str_seconds():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    fd = FeatureDescriptor(
        function=sum_func,
        series_name="EDA",
        window="5s",
        stride="3s",
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit="seconds")
    assert fd.stride == [pd.Timedelta(3, unit="seconds")]
    assert fd.get_required_series() == ["EDA"]
    assert fd.get_nb_output_features() == 1
    assert isinstance(fd.function, FuncWrapper)


def test_simple_feature_descriptor_func_wrapper():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    sum_func_wrapped = FuncWrapper(sum_func)

    fd = FeatureDescriptor(
        function=sum_func_wrapped,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )

    assert fd.series_name == tuple(["EDA"])
    assert fd.window == pd.Timedelta(5, unit="seconds")
    assert fd.stride == [pd.Timedelta(2.5, unit="seconds")]
    assert fd.get_required_series() == ["EDA"]
    assert fd.get_nb_output_features() == 1
    assert isinstance(fd.function, FuncWrapper)


### Test 'error' use-cases


def test_error_function_simple_feature_descriptor():
    invalid_func = []  # Something that is not callable

    with pytest.raises(TypeError):
        _ = FeatureDescriptor(
            function=invalid_func,
            series_name="EDA",
            window="5s",
            stride="2.5s",
        )


def test_error_time_arg_simple_feature_descriptor():
    invalid_stride = pd.to_datetime("13000101", format="%Y%m%d", errors="ignore")

    with pytest.raises(ValueError):
        _ = FeatureDescriptor(
            function=np.sum,
            series_name="EDA",
            window="5s",
            stride=invalid_stride,
        )


def test_error_different_args_simple_feature_descriptor():
    invalid_window, invalid_stride = 15, "5s"  # Invalid combination

    with pytest.raises(TypeError):
        _ = FeatureDescriptor(
            function=np.sum,
            series_name="EDA",
            window=invalid_window,
            stride=invalid_stride,
        )


def test_error_optional_window_but_pass_stride_feature_descriptor():
    with pytest.raises(AssertionError):
        _ = FeatureDescriptor(
            function=np.sum,
            series_name="EDA",
            stride="3s",
            # passes no window
        )


### MultipleFeatureDescriptors


def test_multiple_feature_descriptors():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    mfd = MultipleFeatureDescriptors(
        functions=[sum_func, FuncWrapper(np.max, "amax"), np.mean],
        series_names=["EDA", "TMP"],
        windows=["5s", "7.5s"],
        strides="2.5s",
    )

    assert len(mfd.feature_descriptions) == 3 * 2 * 2

    series_names = [fd.series_name for fd in mfd.feature_descriptions]
    assert set(series_names) == set([tuple(["EDA"]), tuple(["TMP"])])
    assert sum([el == tuple(["EDA"]) for el in series_names]) == 3 * 2
    assert sum([el == tuple(["TMP"]) for el in series_names]) == 3 * 2

    windows = [fd.window for fd in mfd.feature_descriptions]
    assert set(windows) == set([pd.Timedelta(seconds=5), pd.Timedelta(seconds=7.5)])
    assert sum([el == pd.Timedelta(seconds=5) for el in windows]) == 3 * 2
    assert sum([el == pd.Timedelta(seconds=7.5) for el in windows]) == 3 * 2

    strides = flatten([fd.stride for fd in mfd.feature_descriptions])
    assert set(strides) == set([pd.Timedelta(seconds=2.5)])

    functions = [fd.function for fd in mfd.feature_descriptions]
    assert len(set(functions)) == 3
    output_names = [f.output_names for f in functions]
    assert all([len(outputs) == 1 for outputs in output_names])
    output_names = [outputs[0] for outputs in output_names]
    assert set(output_names) == set(["sum_func", "amax", "mean"])
    assert sum([el == "sum_func" for el in output_names]) == 2 * 2
    assert sum([el == "amax" for el in output_names]) == 2 * 2
    assert sum([el == "mean" for el in output_names]) == 2 * 2


def test_multiple_feature_descriptors_optional_stride():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    mfd = MultipleFeatureDescriptors(
        functions=[sum_func, FuncWrapper(np.max, "amax"), np.mean],
        series_names=["EDA", "TMP"],
        windows=["5s", "7.5s"],
        # passes no stride
    )

    assert len(mfd.feature_descriptions) == 3 * 2 * 2

    series_names = [fd.series_name for fd in mfd.feature_descriptions]
    assert set(series_names) == set([tuple(["EDA"]), tuple(["TMP"])])
    assert sum([el == tuple(["EDA"]) for el in series_names]) == 3 * 2
    assert sum([el == tuple(["TMP"]) for el in series_names]) == 3 * 2

    windows = [fd.window for fd in mfd.feature_descriptions]
    assert set(windows) == set([pd.Timedelta(seconds=5), pd.Timedelta(seconds=7.5)])
    assert sum([el == pd.Timedelta(seconds=5) for el in windows]) == 3 * 2
    assert sum([el == pd.Timedelta(seconds=7.5) for el in windows]) == 3 * 2

    strides = [fd.stride for fd in mfd.feature_descriptions]
    assert set(strides) == set([None])

    functions = [fd.function for fd in mfd.feature_descriptions]
    assert len(set(functions)) == 3
    output_names = [f.output_names for f in functions]
    assert all([len(outputs) == 1 for outputs in output_names])
    output_names = [outputs[0] for outputs in output_names]
    assert set(output_names) == set(["sum_func", "amax", "mean"])
    assert sum([el == "sum_func" for el in output_names]) == 2 * 2
    assert sum([el == "amax" for el in output_names]) == 2 * 2
    assert sum([el == "mean" for el in output_names]) == 2 * 2


def test_multiple_feature_descriptors_optional_stride_and_window():
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    mfd = MultipleFeatureDescriptors(
        functions=[sum_func, FuncWrapper(np.max, "amax"), np.mean],
        series_names=["EDA", "TMP"],
        # passes no window,
        # passes no stride
    )

    assert len(mfd.feature_descriptions) == 3 * 2

    series_names = [fd.series_name for fd in mfd.feature_descriptions]
    assert set(series_names) == set([tuple(["EDA"]), tuple(["TMP"])])
    assert sum([el == tuple(["EDA"]) for el in series_names]) == 3
    assert sum([el == tuple(["TMP"]) for el in series_names]) == 3

    windows = [fd.window for fd in mfd.feature_descriptions]
    assert set(windows) == set([None])

    strides = [fd.stride for fd in mfd.feature_descriptions]
    assert set(strides) == set([None])

    functions = [fd.function for fd in mfd.feature_descriptions]
    assert len(set(functions)) == 3
    output_names = [f.output_names for f in functions]
    assert all([len(outputs) == 1 for outputs in output_names])
    output_names = [outputs[0] for outputs in output_names]
    assert set(output_names) == set(["sum_func", "amax", "mean"])
    assert sum([el == "sum_func" for el in output_names]) == 2
    assert sum([el == "amax" for el in output_names]) == 2
    assert sum([el == "mean" for el in output_names]) == 2
