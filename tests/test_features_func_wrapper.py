"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import functools
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from tsflex.features import FuncWrapper

from .utils import dummy_data

## FuncWrapper


def test_simple_numpy_func_wrapper(dummy_data):
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    np_func = FuncWrapper(sum_func)

    assert np_func.output_names == ["sum_func"]
    assert np.isclose(np_func(dummy_data["EDA"]), dummy_data["EDA"].sum())


def test_multi_output_numpy_func_wrapper(dummy_data):
    def mean_std(sig: np.ndarray) -> Tuple[float, float]:
        return np.mean(sig), np.std(sig)

    np_func = FuncWrapper(mean_std, output_names=["mean", "std"])

    assert np_func.output_names == ["mean", "std"]
    mean, std = np_func(dummy_data["TMP"])
    assert np.isclose(mean, dummy_data["TMP"].mean())
    assert np.isclose(std, dummy_data["TMP"].std(), rtol=0.001)


def test_multi_input_numpy_func_wrapper(dummy_data):
    def mean_abs_diff(sig1: np.ndarray, sig2: np.ndarray) -> float:
        return np.mean(np.abs(sig1 - sig2))

    np_func = FuncWrapper(mean_abs_diff)

    assert np_func.output_names == ["mean_abs_diff"]
    res = np_func(dummy_data["EDA"], dummy_data["TMP"])
    assert np.isclose(res, (dummy_data["EDA"] - dummy_data["TMP"]).abs().mean())


def test_kwargs_numpy_func_wrapper(dummy_data):
    def return_arg2(sig: np.ndarray, arg2=5):
        return arg2

    np_func1 = FuncWrapper(return_arg2, output_names="arg2")
    np_func2 = FuncWrapper(return_arg2, output_names="arg2", arg2=10)
    np_func3 = FuncWrapper(return_arg2, arg2=15, output_names="arg2")

    assert np_func1.output_names == ["arg2"]
    assert np_func2.output_names == ["arg2"]
    assert np_func3.output_names == ["arg2"]
    assert np_func1([]) == 5
    assert np_func2([]) == 10
    assert np_func3([]) == 15


def test_series_func_wrapper(dummy_data):
    def max_diff(x: pd.Series):
        return x.index.to_series().diff().dt.total_seconds().max()

    func = FuncWrapper(max_diff, input_type=pd.Series)

    assert func.output_names == ["max_diff"]
    assert func(dummy_data["EDA"]) == 0.25


def test_series_func_wrapper_with_kwargs(dummy_data):
    def max_diff(x: pd.Series, mult=1):
        return x.index.to_series().diff().dt.total_seconds().max() * mult

    func1 = FuncWrapper(max_diff, input_type=pd.Series)
    func2 = FuncWrapper(max_diff, input_type=pd.Series, mult=3, output_names="MAX_DIFF")

    assert func1.output_names == ["max_diff"]
    assert func2.output_names == ["MAX_DIFF"]
    assert func1(dummy_data["EDA"]) == 0.25
    assert func2(dummy_data["EDA"]) == 0.25 * 3


def test_vectorized_func_wrapper(dummy_data):
    func_cols = FuncWrapper(np.max, vectorized=True, axis=0)  # Axis = columns
    func_rows = FuncWrapper(np.max, vectorized=True, axis=1)  # Axis = rows

    assert func_cols.output_names == ["amax"]
    assert func_rows.output_names == ["amax"]
    assert np.allclose(func_cols(dummy_data.values), dummy_data.max().values)
    assert np.allclose(func_rows(dummy_data.values), dummy_data.max(axis=1).values)


def test_functools_support(dummy_data):
    func1 = FuncWrapper(np.quantile, q=0.7)
    func2 = FuncWrapper(functools.partial(np.quantile, q=0.7))

    assert func1.output_names == ["quantile"]
    assert func2.output_names == ["quantile"]
    assert np.allclose(func1(dummy_data.values), func2(dummy_data.values))


def test_error_func_wrapper_wrong_outputnames_type():
    with pytest.raises(TypeError):
        FuncWrapper(np.min, output_names=5)


def test_illegal_func_wrapper_vectorized_wrong_input_type():
    with pytest.raises(AssertionError):
        FuncWrapper(np.min, input_type=pd.Series, vectorized=True, axis=1)
