"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd
import numpy as np

from tsflex.features import NumpyFuncWrapper
from typing import Tuple

from .utils import dummy_data, dataframe_to_series_dict, series_to_series_dict


## NumpyFuncWrapper


def test_simple_numpy_func_wrapper(dummy_data):
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    np_func = NumpyFuncWrapper(sum_func)

    assert np_func.output_names == ["sum_func"]
    assert np.isclose(np_func(dummy_data["EDA"]), dummy_data["EDA"].sum())


def test_multi_output_numpy_func_wrapper(dummy_data):
    def mean_std(sig: np.ndarray) -> Tuple[float, float]:
        return np.mean(sig), np.std(sig)

    np_func = NumpyFuncWrapper(mean_std, output_names=["mean", "std"])

    assert np_func.output_names == ["mean", "std"]
    mean, std = np_func(dummy_data["TMP"])
    assert np.isclose(mean, dummy_data["TMP"].mean())
    assert np.isclose(std, dummy_data["TMP"].std(), rtol=0.001)

def test_multi_input_numpy_func_wrapper(dummy_data):
    def mean_abs_diff(sig1: np.ndarray, sig2: np.ndarray) -> float:
        return np.mean(np.abs(sig1 - sig2))

    np_func = NumpyFuncWrapper(mean_abs_diff)

    assert np_func.output_names == ['mean_abs_diff']
    res = np_func(dummy_data['EDA'], dummy_data['TMP'])
    assert np.isclose(res, (dummy_data['EDA'] - dummy_data['TMP']).abs().mean())

def test_kwargs_numpy_func_wrapper(dummy_data):
    def return_arg2(sig: np.ndarray, arg2=5):
        return arg2
    
    np_func1 = NumpyFuncWrapper(return_arg2, output_names='arg2')
    np_func2 = NumpyFuncWrapper(return_arg2, output_names='arg2', arg2=10)
    np_func3 = NumpyFuncWrapper(return_arg2, arg2=15, output_names='arg2')

    assert np_func1.output_names == ['arg2']
    assert np_func2.output_names == ['arg2']
    assert np_func3.output_names == ['arg2']
    assert np_func1([]) == 5
    assert np_func2([]) == 10
    assert np_func3([]) == 15
