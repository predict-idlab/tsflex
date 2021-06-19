"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pytest
import warnings
import pandas as pd
import numpy as np

from tsflex.features import NumpyFuncWrapper
from tsflex.features import FeatureDescriptor, MultipleFeatureDescriptors
from tsflex.features import FeatureCollection

from pandas.testing import assert_index_equal, assert_frame_equal
from .utils import dummy_data


## FeatureCollection

def test_single_series_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum, series_name="EDA", window='5s', stride='2.5s',
    )
    fc = FeatureCollection(feature_descriptors=fd)

    assert fc.get_required_series() == ["EDA"]

    res_list = fc.calculate(dummy_data, return_df=False, n_jobs=1)
    res_df = fc.calculate(dummy_data, return_df=True, n_jobs=1)

    assert isinstance(res_list, list) & (len(res_list) == 1)
    assert isinstance(res_df, pd.DataFrame)
    assert_frame_equal(res_list[0], res_df)
    freq =  pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, 's')
    stride_s = 2.5; window_s = 5
    assert len(res_df) == (int(len(dummy_data) / (1 / freq)) - window_s) // stride_s
    assert all(res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(2.5, unit='s'))


def test_uneven_sampled_series_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum, series_name="EDA", window='5s', stride='2.5s',
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(FeatureDescriptor(np.min, series_name=("TMP",), window=5, stride=2.5))
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window=5, stride=2.5))

    assert set(fc.get_required_series()) == set(["EDA", "TMP"])
    assert len(fc.get_required_series()) == 2

    # Drop some data to obtain an irregular sampling rate
    inp = dummy_data.drop(np.random.choice(dummy_data.index[1:-1], 500, replace=False))
    
    res_df = fc.calculate(inp, return_df=True, approve_sparsity=True, n_jobs=3)

    assert res_df.shape[1] == 3
    freq =  pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, 's')
    stride_s = 2.5; window_s = 5
    assert len(res_df) == (int(len(dummy_data) / (1 / freq)) - window_s) // stride_s
    assert all(res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(2.5, unit='s'))


def test_warning_uneven_sampled_series_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum, series_name="EDA", window='5s', stride='2.5s',
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(FeatureDescriptor(np.min, series_name=("TMP",), window=5, stride=2.5))

    assert set(fc.get_required_series()) == set(["EDA", "TMP"])
    assert len(fc.get_required_series()) == 2

    # Drop some data to obtain an irregular sampling rate
    inp = dummy_data.drop(np.random.choice(dummy_data.index[1:-1], 500, replace=False))
    
    with warnings.catch_warnings(record=True) as w:
        # Trigger the warning
        res_df = fc.calculate(inp, return_df=True, approve_sparsity=False)
        # Verify the warning
        assert len(w) == 2
        assert all([issubclass(warn.category, RuntimeWarning) for warn in w])
        assert all(['gaps in the time-series' in str(warn) for warn in w])
        # Check the output
        assert res_df.shape[1] == 2
        freq =  pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, 's')
        stride_s = 2.5; window_s = 5
        assert len(res_df) == (int(len(dummy_data) / (1 / freq)) - window_s) // stride_s
        assert all(res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(2.5, unit='s'))


def test_window_idx_single_series_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum, series_name="EDA", window='5s', stride='2.5s',
    )
    fc = FeatureCollection(feature_descriptors=fd)

    assert fc.get_required_series() == ["EDA"]

    res_begin = fc.calculate(dummy_data, return_df=True, window_idx='begin')
    res_end = fc.calculate(dummy_data, return_df=True, window_idx='end')
    res_middle = fc.calculate(dummy_data, return_df=True, window_idx='middle')

    assert np.isclose(res_begin.values, res_end.values).all()
    assert np.isclose(res_begin.values, res_middle.values).all()

    assert res_begin.index[0] == dummy_data.index[0] 
    assert res_end.index[0] == dummy_data.index[0] + pd.to_timedelta(5, unit='s')
    assert res_middle.index[0] == dummy_data.index[0] + pd.to_timedelta(2.5, unit='s')

    for res_df in [res_begin, res_end, res_middle]:
        freq =  pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, 's')
        stride_s = 2.5; window_s = 5
        assert len(res_df) == (int(len(dummy_data) / (1 / freq)) - window_s) // stride_s
        assert all(res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(2.5, unit='s'))



# TODO: test many to one etc.
# TODO: test add MultiplefeatureDescriptors
# TODO: test typeerror of add
