"""Tests for the features functionality."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import os
import random

import dill
import pytest
import math
import warnings
import pandas as pd
import numpy as np

from tsflex.features import FuncWrapper
from tsflex.features import FeatureDescriptor, MultipleFeatureDescriptors
from tsflex.features import FeatureCollection
from tsflex.utils.data import flatten

from pathlib import Path
from pandas.testing import assert_frame_equal
from scipy.stats import linregress
from typing import Tuple, List
from .utils import dummy_data


## FeatureCollection


def test_single_series_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="10s",
        stride="5s",
    )
    fc = FeatureCollection(feature_descriptors=fd)

    assert fc.get_required_series() == ["EDA"]

    res_list = fc.calculate(dummy_data, return_df=False, n_jobs=1)
    res_df = fc.calculate(dummy_data, return_df=True, n_jobs=1)

    assert isinstance(res_list, list) & (len(res_list) == 1)
    assert isinstance(res_df, pd.DataFrame)
    assert_frame_equal(res_list[0], res_df)
    freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
    stride_s = 5
    window_s = 10
    assert len(res_df) == math.ceil(
        (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)
    assert all(res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(5, unit="s"))


def test_single_series_feature_collection_strides(dummy_data):
    stride = "5s"
    fd1 = FeatureDescriptor(np.sum, series_name="EDA", window="10s")
    fd2 = FeatureDescriptor(np.sum, series_name="EDA", window="10s", stride='20s')
    fd3 = FeatureDescriptor(np.sum, series_name="EDA", window="10s", stride=stride)
    fc1 = FeatureCollection(feature_descriptors=fd1)
    fc2 = FeatureCollection(feature_descriptors=fd2)
    fc3 = FeatureCollection(feature_descriptors=fd3)

    assert fc1.get_required_series() == fc2.get_required_series()
    assert fc1.get_required_series() == fc3.get_required_series()

    res1 = fc1.calculate(dummy_data, stride=stride, return_df=False, n_jobs=1)
    res2 = fc2.calculate(dummy_data, stride=stride, return_df=False, n_jobs=1)
    res3 = fc3.calculate(dummy_data, return_df=False, n_jobs=1)

    assert (len(res1) == 1) & (len(res2) == 1) & (len(res3) == 1)

    assert_frame_equal(res1[0], res2[0])
    assert_frame_equal(res1[0], res3[0])


def test_single_series_feature_collection_sequence_setpoints(dummy_data):
    dummy_data = dummy_data.reset_index(drop=True)
    setpoints = [0, 5, 7, 10]
    fd1 = FeatureDescriptor(np.sum, series_name="EDA", window=10)
    fd2 = FeatureDescriptor(np.sum, series_name="EDA", window=10, stride=20)
    fc1 = FeatureCollection(feature_descriptors=fd1)
    fc2 = FeatureCollection(feature_descriptors=fd2)

    assert fc1.get_required_series() == fc2.get_required_series()

    res1 = fc1.calculate(dummy_data, setpoints=setpoints, window_idx="begin")
    res2 = fc2.calculate(dummy_data, setpoints=setpoints, window_idx="begin")

    assert (len(res1) == 1) & (len(res2) == 1)
    assert (len(res1[0]) == 4) & (len(res2[0]) == 4)

    assert_frame_equal(res1[0], res2[0])
    assert all(res1[0].index.values == setpoints)


def test_single_series_feature_collection_timestamp_setpoints(dummy_data):
    setpoints = [0, 5, 7, 10]
    setpoints = dummy_data.index[setpoints].values
    fd1 = FeatureDescriptor(np.sum, series_name="EDA", window="10s")
    fd2 = FeatureDescriptor(np.sum, series_name="EDA", window="10s", stride='20s')
    fc1 = FeatureCollection(feature_descriptors=fd1)
    fc2 = FeatureCollection(feature_descriptors=fd2)

    assert fc1.get_required_series() == fc2.get_required_series()

    res1 = fc1.calculate(dummy_data, setpoints=setpoints, window_idx="begin")
    res2 = fc2.calculate(dummy_data, setpoints=setpoints, window_idx="begin")

    assert (len(res1) == 1) & (len(res2) == 1)
    assert (len(res1[0]) == 4) & (len(res2[0]) == 4)

    assert_frame_equal(res1[0], res2[0])
    assert all(res1[0].index.values == setpoints)


def test_single_series_feature_collection_sequence_setpoints_datatypes():
    s = pd.Series(np.arange(20), name="dummy")
    setpoints_list = [0, 3, 5, 6, 8]

    fc = FeatureCollection(
        FeatureDescriptor(np.min, "dummy", 5)
    )

    # On a list
    setpoints = setpoints_list
    assert isinstance(setpoints, list)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])

    # On a numpy array
    setpoints = np.array(setpoints_list)
    assert isinstance(setpoints, np.ndarray)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])

    # On a pandas series
    setpoints = pd.Series(setpoints_list)
    assert isinstance(setpoints, pd.Series)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])

    # On a pandas dataframe
    setpoints = pd.DataFrame(setpoints_list)
    assert isinstance(setpoints, pd.DataFrame)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])

    # On a pandas index
    setpoints = pd.Index(setpoints_list)
    assert isinstance(setpoints, pd.Index)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])


def test_single_series_feature_collection_timestamp_setpoints_datatypes():
    s = pd.Series(np.arange(20), name="dummy")
    s.index = pd.date_range("2021-08-09", freq="1h", periods=20)
    setpoints_list = [0, 3, 5, 6, 8]

    fc = FeatureCollection(
        FeatureDescriptor(np.min, "dummy", "1h")
    )

    # On a list
    setpoints = [s.index[idx] for idx in setpoints_list]
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True, n_jobs=0)
    assert np.all(res.index == s.index[setpoints_list])

    # On a numpy array
    setpoints = s.index[setpoints_list].values
    assert isinstance(setpoints, np.ndarray)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])

    # On a pandas series
    setpoints = pd.Series(s.index[setpoints_list])
    assert isinstance(setpoints, pd.Series)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])

    # On a pandas dataframe
    setpoints = pd.DataFrame(s.index[setpoints_list])
    assert isinstance(setpoints, pd.DataFrame)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])

    # On a pandas index
    setpoints = s.index[setpoints_list]
    assert isinstance(setpoints, pd.Index)
    res = fc.calculate(s, setpoints=setpoints, window_idx="begin", return_df=True)
    assert np.all(res.index == s.index[setpoints_list])


def test_sequence_setpoints_not_sorted():
    s = pd.Series(np.arange(20), name="dummy")
    setpoints = [0, 5, 3]

    fc = FeatureCollection(
        FeatureDescriptor(np.min, "dummy", 5)
    )
    res = fc.calculate(s, setpoints=setpoints, return_df=True, window_idx="begin")
    assert all(res.index == setpoints)


def test_time_setpoints_not_sorted():
    s = pd.Series(np.arange(20), name="dummy")
    s.index = pd.date_range("2021-08-09", freq="1h", periods=20)
    setpoints = s.index[[0, 5, 3]]

    fc = FeatureCollection(
        FeatureDescriptor(np.min, "dummy", "1h")
    )
    res = fc.calculate(s, setpoints=setpoints, return_df=True, window_idx="begin")
    assert all(res.index == setpoints)


def test_uneven_sampled_series_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="10s",
        stride="16s",
    )
    fc = FeatureCollection(feature_descriptors=fd)
    with pytest.raises(ValueError):
        fc.add(FeatureDescriptor(np.min, series_name=("TMP",), window="10", stride="6"))
    with pytest.raises(ValueError):
        fc.add(FeatureDescriptor(np.min, series_name=("TMP",), window="5s", stride="6"))
    with pytest.raises(ValueError):
        fc.add(FeatureDescriptor(np.min, series_name=("TMP",), window="5", stride="6s"))

    fc.add(FeatureDescriptor(np.min, series_name=("TMP",), window="10s", stride="16s"))
    fc.add(FeatureDescriptor(np.min, series_name=("EDA",), window="10s", stride="16s"))

    assert set(fc.get_required_series()) == set(["EDA", "TMP"])
    assert len(fc.get_required_series()) == 2

    # Drop some data to obtain an irregular sampling rate
    inp = dummy_data.drop(np.random.choice(dummy_data.index[1:-1], 500, replace=False))

    res_df = fc.calculate(inp, return_df=True, approve_sparsity=True, n_jobs=3)

    assert res_df.shape[1] == 3
    freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
    stride_s = 16
    window_s = 10
    assert len(res_df) == math.ceil(
        (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)
    assert all(
        res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(stride_s, unit="s")
    )


def test_warning_uneven_sampled_series_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )
    fc = FeatureCollection(feature_descriptors=fd)
    fc.add(FeatureDescriptor(np.min, series_name=("TMP",), window="5s", stride="2.5s"))

    assert set(fc.get_required_series()) == set(["EDA", "TMP"])
    assert len(fc.get_required_series()) == 2
    # Drop some data to obtain an irregular sampling rate
    inp = dummy_data.drop(np.random.choice(dummy_data.index[1:-1], 500, replace=False))

    with warnings.catch_warnings(record=True) as w:
        # Trigger the warning
        # Note -> for some (yet unkknown) reason, the warning's aren't caught anymore
        # when using multiprocess (they are thrown nevertheless!), so we changed
        # n_jobs=1
        res_df = fc.calculate(inp, return_df=True, n_jobs=1, approve_sparsity=False)
        # Verify the warning
        assert len(w) == 2
        assert all([issubclass(warn.category, RuntimeWarning) for warn in w])
        assert all(["gaps in the sequence" in str(warn) for warn in w])
        # Check the output
        assert res_df.shape[1] == 2
        freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
        stride_s = 2.5
        window_s = 5
        assert len(res_df) == math.ceil(
            (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)
        assert all(
            res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(2.5, unit="s")
        )


def test_featurecollection_repr(dummy_data):
    def corr(s1, s2):
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        return np.corrcoef(s1, s2)[0][-1].astype(s1.dtype)

    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(
                function=FuncWrapper(func=corr, output_names="corrcoef"),
                series_name=("EDA", "TMP"),
                window="30s",
                stride="30s",
            ),
        ]
    )
    fc_str: str = fc.__repr__()
    assert "EDA|TMP" in fc_str
    assert (
        fc_str
        == "EDA|TMP: (\n\twin: 30s   : [\n\t\tFeatureDescriptor - func: FuncWrapper(corr, ['corrcoef'], {})    stride: ['30s'],\n\t]\n)\n"
    )

    out = fc.calculate(dummy_data, n_jobs=1, return_df=True)
    assert out.columns[0] == "EDA|TMP__corrcoef__w=30s"

    out = fc.calculate(dummy_data, n_jobs=None, return_df=True)
    assert out.columns[0] == "EDA|TMP__corrcoef__w=30s"


def test_window_idx_single_series_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="12.5s",
    )
    fc = FeatureCollection(feature_descriptors=fd)

    assert fc.get_required_series() == ["EDA"]

    res_begin = fc.calculate(dummy_data, return_df=True, n_jobs=0, window_idx="begin")
    res_end = fc.calculate(dummy_data, return_df=True, n_jobs=0, window_idx="end")
    res_middle = fc.calculate(dummy_data, return_df=True, n_jobs=0, window_idx="middle")
    assert np.isclose(res_begin.values, res_end.values).all()
    assert np.isclose(res_begin.values, res_middle.values).all()

    res_begin = fc.calculate(
        dummy_data, return_df=True, n_jobs=None, window_idx="begin"
    )
    res_end = fc.calculate(dummy_data, return_df=True, n_jobs=None, window_idx="end")
    res_middle = fc.calculate(
        dummy_data, return_df=True, n_jobs=None, window_idx="middle"
    )

    with pytest.raises(Exception):
        res_not_existing = fc.calculate(
            dummy_data, n_jobs=0, return_df=True, window_idx="somewhere"
        )

    assert np.isclose(res_begin.values, res_end.values).all()
    assert np.isclose(res_begin.values, res_middle.values).all()

    assert res_begin.index[0] == dummy_data.index[0]
    assert res_end.index[0] == dummy_data.index[0] + pd.to_timedelta(5, unit="s")
    # 2.5 -> refers to window / 2
    assert res_middle.index[0] == dummy_data.index[0] + pd.to_timedelta(2.5, unit="s")

    for res_df in [res_begin, res_end, res_middle]:
        freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
        stride_s = 12.5
        window_s = 5
        assert len(res_df) == math.ceil(
            (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)
        assert all(
            res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(12.5, unit="s")
        )


def test_multiplefeaturedescriptors_feature_collection(dummy_data):
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    mfd = MultipleFeatureDescriptors(
        functions=[sum_func, FuncWrapper(np.max), np.min],
        series_names=["EDA", "TMP"],
        windows=["5s", "7.5s"],
        strides="2.5s",
    )
    fc = FeatureCollection(feature_descriptors=mfd)

    assert set(fc.get_required_series()) == set(["EDA", "TMP"])
    assert len(fc.get_required_series()) == 2

    res_list = fc.calculate(dummy_data, return_df=False, n_jobs=6)
    res_df = fc.calculate(dummy_data, return_df=True, n_jobs=6)

    assert (len(res_list) == 3 * 2 * 2) & (res_df.shape[1] == 3 * 2 * 2)
    res_list_names = [res.columns.values[0] for res in res_list]
    assert set(res_list_names) == set(res_df.columns)
    expected_output_names = [
        [
            f"{sig}__sum_func__w=5s",
            f"{sig}__sum_func__w=7.5s",
            f"{sig}__amax__w=5s",
            f"{sig}__amax__w=7.5s",
            f"{sig}__amin__w=5s",
            f"{sig}__amin__w=7.5s",
        ]
        for sig in ["EDA", "TMP"]
    ]
    # Flatten
    expected_output_names = expected_output_names[0] + expected_output_names[1]
    assert set(res_df.columns) == set(expected_output_names)

    # No NaNs when returning a list of calculated featured
    assert all([~res.isna().values.any() for res in res_list])
    # NaNs when merging to a df (for  some cols)
    assert all([res_df[col].isna().any() for col in res_df.columns if "w=7.5s" in col])
    assert all([~res_df[col].isna().any() for col in res_df.columns if "w=5s" in col])

    stride_s = 2.5
    window_s = 5
    freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
    expected_length = math.ceil(
        (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)
    assert all(
        [
            len(res) == expected_length - 1
            for res in res_list
            if "w=7.5s" in res.columns.values[0]
        ]
    )
    assert all(
        [
            len(res) == expected_length
            for res in res_list
            if "w=5s" in res.columns.values[0]
        ]
    )
    assert len(res_df) == expected_length


def test_multiplefeaturedescriptors_feature_collection_strides(dummy_data):
    stride= "2.5s"
    mfd1 = MultipleFeatureDescriptors([np.max, np.min], ["EDA", "TMP"], ["5s", "7.5s"])
    mfd2 = MultipleFeatureDescriptors([np.max, np.min], ["EDA", "TMP"], ["5s", "7.5s"], strides=["5s"])#, "10s"])  # TODO: list of strides supporten...
    mfd3 = MultipleFeatureDescriptors([np.max, np.min], ["EDA", "TMP"], ["5s", "7.5s"], strides=stride)
    fc1 = FeatureCollection(mfd1)
    fc2 = FeatureCollection(mfd2)
    fc3 = FeatureCollection(mfd3)

    assert fc1.get_required_series() == fc2.get_required_series()
    assert fc1.get_required_series() == fc3.get_required_series()

    res1 = fc1.calculate(dummy_data, stride=stride, return_df=True, n_jobs=0)
    res2 = fc2.calculate(dummy_data, stride=stride, return_df=True, n_jobs=0)
    res3 = fc3.calculate(dummy_data, return_df=True, n_jobs=0)

    assert_frame_equal(res1, res2)
    assert_frame_equal(res1, res3)


def test_featurecollection_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )
    fc = FeatureCollection(FeatureCollection(feature_descriptors=fd))

    assert fc.get_required_series() == ["EDA"]

    res_list = fc.calculate(dummy_data, return_df=False, n_jobs=1)
    res_df = fc.calculate(dummy_data, return_df=True, n_jobs=1)

    assert isinstance(res_list, list) & (len(res_list) == 1)
    assert isinstance(res_df, pd.DataFrame)
    assert_frame_equal(res_list[0], res_df)
    freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
    stride_s = 2.5
    window_s = 5
    assert len(res_df) == math.ceil(
        (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)
    assert all(res_df.index[1:] - res_df.index[:-1] == pd.to_timedelta(2.5, unit="s"))


def test_feature_collection_column_sorted(dummy_data):
    fc = FeatureCollection(
        MultipleFeatureDescriptors(
            functions=[np.max, np.min, len, np.sum, np.median, np.mean, np.std],
            series_names="EDA",
            windows=["5s", "30s", "2min"],
            strides="30s",
        )
    )
    df_eda = dummy_data["EDA"].first('5min')
    out_cols = fc.calculate(df_eda, return_df=True, n_jobs=None).columns.values

    for _ in range(10):
        assert all(out_cols == fc.calculate(df_eda, return_df=True).columns.values)


def test_featurecollection_reduce(dummy_data):
    fc = FeatureCollection(
        MultipleFeatureDescriptors(
            functions=[np.max, np.min, np.std, np.sum],
            series_names="EDA",
            windows=["5s", "30s", "1min"],
            strides="30s",
        )
    )
    df_feat_tot = fc.calculate(data=dummy_data, return_df=True, show_progress=True)

    for _ in range(5):
        col_subset = random.sample(
            list(df_feat_tot.columns), random.randint(1, len(df_feat_tot.columns))
        )
        fc_reduced = fc.reduce(col_subset)
        fc_reduced.calculate(dummy_data)
        for fd in flatten(fc._feature_desc_dict.values()):
            assert np.all(fd.stride == [pd.Timedelta(30, unit="s")])

    # also test the reduce function on a single column
    fc_reduced = fc.reduce(random.sample(list(df_feat_tot.columns), 1))
    fc_reduced.calculate(dummy_data)

    # should also work when fc is deleted
    del fc
    fc_reduced.calculate(dummy_data)


def test_featurecollection_reduce_multiple_strides(dummy_data):
    fc = FeatureCollection(
        MultipleFeatureDescriptors(
            functions=[np.max, np.min, np.std, np.sum],
            series_names="EDA",
            windows=["5s", "30s", "1min"],
            strides=["30s", "45s"],
        )
    )
    df_feat_tot = fc.calculate(data=dummy_data, return_df=True, show_progress=True)

    for _ in range(5):
        col_subset = random.sample(
            list(df_feat_tot.columns), random.randint(1, len(df_feat_tot.columns))
        )
        fc_reduced = fc.reduce(col_subset)
        fc_reduced.calculate(dummy_data)
        for fd in flatten(fc._feature_desc_dict.values()):
            assert np.all(fd.stride == [pd.Timedelta(30, unit="s"), pd.Timedelta(45, unit="s")])

    # also test the reduce function on a single column
    fc_reduced = fc.reduce(random.sample(list(df_feat_tot.columns), 1))
    fc_reduced.calculate(dummy_data)

    # should also work when fc is deleted
    del fc
    fc_reduced.calculate(dummy_data)


def test_featurecollection_reduce_no_stride(dummy_data):
    fc = FeatureCollection(
        MultipleFeatureDescriptors(
            functions=[np.max, np.min, np.std, np.sum],
            series_names="EDA",
            windows=["5s", "30s", "1min"],
        )
    )
    df_feat_tot = fc.calculate(data=dummy_data, stride="45s", return_df=True, show_progress=True)

    for _ in range(5):
        col_subset = random.sample(
            list(df_feat_tot.columns), random.randint(1, len(df_feat_tot.columns))
        )
        fc_reduced = fc.reduce(col_subset)
        fc_reduced.calculate(dummy_data, stride="45s")
        for fd in flatten(fc._feature_desc_dict.values()):
            assert fd.stride == None

    # also test the reduce function on a single column
    fc_reduced = fc.reduce(random.sample(list(df_feat_tot.columns), 1))
    fc_reduced.calculate(dummy_data, stride="45s")

    # should also work when fc is deleted
    del fc
    fc_reduced.calculate(dummy_data, stride="45s")


def test_featurecollection_numeric_reduce(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=[
            MultipleFeatureDescriptors(
                windows=[240, 480, 1000],
                strides=240,
                functions=[np.mean, np.min, np.max, np.std],
                series_names=["TMP", "EDA"],
            )
        ]
    )
    df_tmp = dummy_data["TMP"].reset_index(drop=True)
    df_eda = dummy_data["EDA"].reset_index(drop=True)
    out = fc.calculate([df_tmp, df_eda], window_idx="end", return_df=True)

    n_retain = 8
    fc_reduced = fc.reduce(np.random.choice(out.columns, size=n_retain, replace=False))
    out_2 = fc_reduced.calculate([df_tmp, df_eda], return_df=True)
    assert out_2.shape[1] == n_retain


def test_featurecollection_reduce_multiple_feat_output(dummy_data):
    def get_stats(series: np.ndarray):
        return np.min(series), np.max(series)

    fd = FeatureDescriptor(
        function=FuncWrapper(get_stats, output_names=["min", "max"]),
        series_name="EDA",
        window="5s",
        stride="5s",
    )

    fc = FeatureCollection(
        [
            MultipleFeatureDescriptors(
                functions=[np.std, np.sum],
                series_names="EDA",
                windows=["5s", "30s", "1min"],
                strides="5s",
            ),
            fd,
        ]
    )
    # df_feat_tot = fc.calculate(data=dummy_data, return_df=True, show_progress=True)

    fc_reduce = fc.reduce(feat_cols_to_keep=["EDA__min__w=5s"])
    del fd
    fc_reduce.calculate(dummy_data)


def test_featurecollection_error_val(dummy_data):
    fd = FeatureDescriptor(
        function=np.max,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )
    fc = FeatureCollection(FeatureCollection(feature_descriptors=fd))

    eda_data = dummy_data["EDA"].dropna()
    eda_data[2: 1 + 25 * 4] = None  # Leave gap of 25 s
    eda_data = eda_data.dropna()
    assert eda_data.isna().any() == False
    assert (eda_data.index[1:] - eda_data.index[:-1]).max() == pd.Timedelta("25 s")

    with pytest.raises(Exception):
        fc.calculate(eda_data, return_df=True, approve_sparsity=True)


def test_featurecollection_error_val_multiple_outputs(dummy_data):
    def get_stats(series: np.ndarray):
        return np.min(series), np.max(series)

    fd = FeatureDescriptor(
        function=FuncWrapper(get_stats, output_names=["min", "max"]),
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )
    fc = FeatureCollection(FeatureCollection(feature_descriptors=fd))

    eda_data = dummy_data["EDA"].dropna()
    eda_data[2: 1 + 25 * 4] = None  # Leave gap of 25 s
    eda_data = eda_data.dropna()
    assert eda_data.isna().any() == False
    assert (eda_data.index[1:] - eda_data.index[:-1]).max() == pd.Timedelta("25 s")

    with pytest.raises(Exception):
        fc.calculate(eda_data, return_df=True, approve_sparsity=True)


def test_feature_collection_invalid_series_names(dummy_data):
    fd = FeatureDescriptor(
        function=FuncWrapper(np.min, output_names=["min"]),
        series_name="EDA__col",  # invalid name, no '__' allowed
        window="10s",
        stride="5s",
    )

    with pytest.raises(Exception):
        fc = FeatureCollection(feature_descriptors=fd)

    fd = FeatureDescriptor(
        function=FuncWrapper(np.min, output_names=["min"]),
        series_name="EDA|col",  # invalid name, no '|' allowed
        window="10s",
        stride="5s",
    )

    with pytest.raises(Exception):
        fc = FeatureCollection(feature_descriptors=fd)


def test_feature_collection_invalid_feature_output_names(dummy_data):
    fd = FeatureDescriptor(
        function=FuncWrapper(np.max, output_names=["max|feat"]),
        series_name="EDA",
        window="10s",
        stride="5s",
    )

    # this should work, no error should be raised
    fc = FeatureCollection(feature_descriptors=fd)

    fd = FeatureDescriptor(
        function=FuncWrapper(np.max, output_names=["max__feat"]),
        # invalid output_name, no '__' allowed
        series_name="EDA",
        window="10s",
        stride="5s",
    )

    with pytest.raises(Exception):
        fc = FeatureCollection(feature_descriptors=fd)


### Test various feature descriptor functions


def test_one_to_many_feature_collection(dummy_data):
    def quantiles(sig: pd.Series) -> Tuple[float, float, float]:
        return np.quantile(sig, q=[0.1, 0.5, 0.9])

    q_func = FuncWrapper(quantiles, output_names=["q_0.1", "q_0.5", "q_0.9"])
    fd = FeatureDescriptor(q_func, series_name="EDA", window="5s", stride="2.5s")
    fc = FeatureCollection(fd)

    res_df = fc.calculate(dummy_data, return_df=True)
    assert res_df.shape[1] == 3
    freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
    stride_s = 2.5
    window_s = 5
    assert len(res_df) == math.ceil(
        (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)

    expected_output_names = [
        "EDA__q_0.1__w=5s",
        "EDA__q_0.5__w=5s",
        "EDA__q_0.9__w=5s",
    ]
    assert set(res_df.columns.values) == set(expected_output_names)
    assert (res_df[expected_output_names[0]] != res_df[expected_output_names[1]]).any()
    assert (res_df[expected_output_names[0]] != res_df[expected_output_names[2]]).any()


def test_many_to_one_feature_collection(dummy_data):
    def abs_mean_diff(sig1: pd.Series, sig2: pd.Series) -> float:
        # Note that this func only works when sig1 and sig2 have the same length
        return np.mean(np.abs(sig1 - sig2))

    fd = FeatureDescriptor(
        abs_mean_diff, series_name=("EDA", "TMP"), window="5s", stride="2.5s"
    )
    fc = FeatureCollection(fd)

    assert set(fc.get_required_series()) == set(["EDA", "TMP"])

    res_df = fc.calculate(dummy_data, return_df=True)
    assert res_df.shape[1] == 1
    freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
    stride_s = 2.5
    window_s = 5
    assert len(res_df) == math.ceil(
        (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)

    expected_output_name = "EDA|TMP__abs_mean_diff__w=5s"
    assert res_df.columns.values[0] == expected_output_name


def test_many_to_many_feature_collection(dummy_data):
    def quantiles_abs_diff(
            sig1: pd.Series, sig2: pd.Series
    ) -> Tuple[float, float, float]:
        return np.quantile(np.abs(sig1 - sig2), q=[0.1, 0.5, 0.9])

    q_func = FuncWrapper(
        quantiles_abs_diff,
        output_names=["q_0.1_abs_diff", "q_0.5_abs_diff", "q_0.9_abs_diff"],
    )
    fd = FeatureDescriptor(
        q_func, series_name=("EDA", "TMP"), window="5s", stride="13.5s"
    )
    fc = FeatureCollection(fd)

    assert set(fc.get_required_series()) == set(["EDA", "TMP"])

    res_df = fc.calculate(dummy_data, return_df=True)
    assert res_df.shape[1] == 3
    freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
    stride_s = 13.5
    window_s = 5
    assert len(res_df) == math.ceil(
        (int(len(dummy_data) / (1 / freq)) - window_s) / stride_s)

    expected_output_names = [
        "EDA|TMP__q_0.1_abs_diff__w=5s",
        "EDA|TMP__q_0.5_abs_diff__w=5s",
        "EDA|TMP__q_0.9_abs_diff__w=5s",
    ]
    assert set(res_df.columns.values) == set(expected_output_names)
    assert (res_df[expected_output_names[0]] != res_df[expected_output_names[1]]).any()
    assert (res_df[expected_output_names[0]] != res_df[expected_output_names[2]]).any()


def test_cleared_pools_when_feature_error(dummy_data):
    def mean_func(s: np.ndarray):
        assert 0 == 1  # make the feature function throw an error
        return np.mean(s)

    fc = FeatureCollection(
        MultipleFeatureDescriptors(
            mean_func, ["EDA", "ACC_x"], ["30s", "45s", "1min", "2min"], "15s"
        )
    )

    for n_jobs in [0, None]:
        with pytest.raises(Exception):
            out = fc.calculate(dummy_data, return_df=True, n_jobs=n_jobs)

    # Now fix the error in the feature function & make sure that pools are cleared,
    # i.e., the same error is not thrown again.
    def mean_func(s: np.ndarray):
        return np.mean(s)

    fc = FeatureCollection(
        MultipleFeatureDescriptors(
            mean_func, ["EDA", "ACC_x"], ["30s", "45s", "1min", "2min"], "15s"
        )
    )

    for n_jobs in [0, None]:
        out = fc.calculate(dummy_data, return_df=True, n_jobs=n_jobs)
        assert out.shape[0] > 0
        assert out.shape[1] == 2 * 4


def test_series_funcs(dummy_data):
    def min_max_time_diff(x: pd.Series, mult=1):
        diff = x.index.to_series().diff().dt.total_seconds()  # .max()
        return diff.min() * mult, diff.max() * mult

    def time_diff(x: pd.Series):
        return (x.index[-1] - x.index[0]).total_seconds()

    def linear_trend_timewise(x):
        """
        Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
        length of the time series minus one.
        This feature uses the index of the time series to fit the model, which must be of a datetime
        dtype.
        The parameters control which of the characteristics are returned.
        Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
        linregress for more information.

        :param x: the time series to calculate the feature of. The index must be datetime.
        :type x: pandas.Series

        :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
        :type param: list

        :return: the different feature values
        :return type: list
        """
        ix = x.index

        # Get differences between each timestamp and the first timestamp in seconds.
        # Then convert to hours and reshape for linear regression
        times_seconds = (ix - ix[0]).total_seconds()
        times_hours = np.asarray(times_seconds / float(3600))

        linReg = linregress(times_hours, x.values)
        return linReg.slope, linReg.intercept, linReg.rvalue

    fc = FeatureCollection(
        MultipleFeatureDescriptors(
            functions=[
                np.mean,
                np.sum,
                len,
                FuncWrapper(
                    min_max_time_diff,
                    input_type=pd.Series,
                    output_names=["min_time_diff", "max_time_diff"],
                    mult=3,
                ),
                FuncWrapper(
                    linear_trend_timewise,
                    input_type=pd.Series,
                    output_names=[
                        "timewise_regr_slope",
                        "timewise_regr_intercept",
                        "timewise_regr_r_value",
                    ],
                ),
                FuncWrapper(time_diff, input_type=pd.Series),
                FuncWrapper(np.max, input_type=np.array),
            ],
            series_names=["EDA", "TMP"],
            windows="5s",
            strides="2.5s",
        )
    )

    assert set(fc.get_required_series()) == set(["EDA", "TMP"])
    downscale_factor = 20
    res_df = fc.calculate(
        dummy_data[: int(len(dummy_data) / downscale_factor)], return_df=True
    )
    # Note: testing this single-threaded allows the code-cov to fire
    res_df_2 = fc.calculate(
        dummy_data[: int(len(dummy_data) / downscale_factor)],
        return_df=True,
        n_jobs=1,
        show_progress=True,
    )
    assert res_df.shape[1] == 2 * 10
    freq = pd.to_timedelta(pd.infer_freq(dummy_data.index)) / np.timedelta64(1, "s")
    stride_s = 2.5
    window_s = 5
    assert len(res_df) == math.ceil(
        (int(len(dummy_data) / downscale_factor / (1 / freq)) - window_s) / stride_s
    )

    assert "EDA__min_time_diff__w=5s" in res_df.columns
    assert "EDA__amax__w=5s" in res_df.columns
    assert all(
        res_df["EDA__min_time_diff__w=5s"]
        == res_df["EDA__max_time_diff__w=5s"]
    )
    assert all(res_df["EDA__min_time_diff__w=5s"] == 0.25 * 3)


def test_categorical_funcs():
    categories = ["a", "b", "c", "another_category", 12]
    categorical_data = pd.Series(
        data=np.random.choice(categories, 1000),
        index=pd.date_range("2021-07-01", freq="1h", periods=1000),
    ).rename("cat")

    # drop some data, as we don't make frequency assumptions
    categorical_data = categorical_data.drop(
        np.random.choice(categorical_data.index, 200, replace=False)
    )

    def count_categories(arr, categories):
        return [sum(arr.astype(str) == str(cat)) for cat in categories]

    cat_count = FuncWrapper(
        func=count_categories,
        output_names=["count-" + str(cat) for cat in categories],
        # kwargs
        categories=categories,
    )

    # construct the collection in which you add all your features
    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(
                function=cat_count, series_name="cat", window="1day", stride="12hours"
            )
        ]
    )

    for n_jobs in [0, None]:
        out = fc.calculate(
            data=categorical_data, approve_sparsity=True, n_jobs=n_jobs, return_df=True
        )
        for c in categories:
            assert f"cat__count-{str(c)}__w=1D" in out.columns


def test_time_based_features():
    # create a time column
    time_value_series = (
        pd.Series(
            index=pd.date_range("2021-07-01", freq="1h", periods=1000), dtype=object
        )
            .index.to_series()
            .rename("time")
    )

    # drop some data, as we don't make frequency assumptions
    time_value_series = time_value_series.drop(
        np.random.choice(time_value_series.index, 250, replace=False)
    )

    def std_hour(time_arr):
        # calcualtes the std in seconds
        if time_arr.shape[0] <= 3:
            return np.NaN
        return np.std(
            np.diff(time_arr).astype("timedelta64[us]").astype(np.int64)
            / (60 * 60 * 1e6)
        )

    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(
                function=std_hour,
                series_name="time",
                window="6 hours",
                stride="4 hours",
            )
        ]
    )
    out = fc.calculate(
        data=time_value_series, approve_sparsity=True, n_jobs=1, return_df=True
    )
    assert out.columns[0] == "time__std_hour__w=6h"

    out = fc.calculate(
        data=time_value_series, approve_sparsity=True, n_jobs=None, return_df=True
    )
    assert out.columns[0] == "time__std_hour__w=6h"


def test_pass_by_value(dummy_data):
    def try_change_view(series_view: np.ndarray):
        series_view[:5] = 0  # update the view -> error!
        return np.mean(series_view)

    fc_gsr = FeatureCollection(
        [
            FeatureDescriptor(
                try_change_view,
                "EDA",
                "30s",
                "15s",
            )
        ]
    )

    for n_jobs in [0, None]:
        with pytest.raises(Exception):
            out = fc_gsr.calculate(dummy_data, return_df=True, n_jobs=n_jobs)


def test_datatype_retention(dummy_data):
    for dtype in [np.float16, np.float32, np.int64, np.int32]:

        def mean_dtype(series_view: np.ndarray):
            return np.mean(series_view).astype(dtype)

        fc_gsr = FeatureCollection(
            [
                FeatureDescriptor(
                    mean_dtype,
                    "EDA",
                    "30s",
                    "15s",
                )
            ]
        )
        for n_jobs in [0, 1, 2, None]:
            print(dtype, n_jobs)
            out = fc_gsr.calculate(dummy_data, return_df=True, n_jobs=n_jobs)
            assert out.values.dtype == dtype


### Test the various input data types combinations

def test_time_based_features_sequence_based_data_error(dummy_data):
    df_eda = dummy_data['EDA'].reset_index()
    df_tmp = dummy_data['TMP'].reset_index()

    fs = 4  # The sample frequency in Hz
    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(np.min, 'EDA', f'{250}s', f'{75}s'),
            FeatureDescriptor(np.min, 'TMP', 250 * fs, 75 * fs)
        ]
    )

    # cannot use time-based win-stride configurations on sequence based data
    with pytest.raises(RuntimeError):
        fc.calculate([df_eda, df_tmp])

    with pytest.raises(RuntimeError):
        fc.calculate([df_eda, df_tmp], n_jobs=0)


def test_mixed_featuredescriptors_time_data(dummy_data):
    df_eda = dummy_data['EDA']
    df_tmp = dummy_data['TMP']

    fs = 4  # The sample frequency in Hz
    with warnings.catch_warnings(record=True) as w:
        # generate the warning by adding mixed FeatureDescriptors
        fc = FeatureCollection(
            feature_descriptors=[
                # same data range -> so when we perform an outer merge we do not suspect a
                # nan error
                FeatureDescriptor(np.min, 'EDA', f'{250}s', f'{75}s'),
                FeatureDescriptor(np.min, 'EDA', 250 * fs, 75 * fs)
            ]
        )
        assert len(w) == 1
        assert all([issubclass(warn.category, RuntimeWarning) for warn in w])
        assert all(
            ["There are multiple FeatureDescriptor window-stride datatypes" in str(warn)
             for warn in w])

    with warnings.catch_warnings(record=True) as w:
        # generate the warning by adding mixed FeatureDescriptors
        fc.add(FeatureDescriptor(np.std, 'EDA', 250 * fs, 75 * fs))
        assert len(w) == 1
        assert all([issubclass(warn.category, RuntimeWarning) for warn in w])
        assert all(
            ["There are multiple FeatureDescriptor window-stride datatypes" in str(warn)
             for warn in w])

    out = fc.calculate([df_eda, df_tmp], return_df=True)
    assert all(out.notna().sum(axis=0))  # TODO what do we want to assert here?


### Test vectorized features

def test_basic_vectorized_features(dummy_data):
    fs = 4  # The sample frequency in Hz
    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(np.max, "EDA", 250 * fs, 75 * fs),
            FeatureDescriptor(
                FuncWrapper(np.max, output_names="max_", vectorized=True, axis=-1),
                "EDA", 250 * fs, 75 * fs,
            )
        ]
    )
    res = fc.calculate(dummy_data)

    assert len(res) == 2
    assert (len(res[0]) > 1) and (len(res[1]) > 1)
    assert np.all(res[0].index == res[1].index)
    assert np.all(res[0].values == res[1].values)


def test_time_based_vectorized_features(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(np.max, "EDA", "5min", "3min"),
            FeatureDescriptor(
                FuncWrapper(np.max, output_names="max_", vectorized=True, axis=-1),
                "EDA", "5min", "3min",
            )
        ]
    )
    res = fc.calculate(dummy_data)

    assert len(res) == 2
    assert (len(res[0]) > 1) and (len(res[1]) > 1)
    assert np.all(res[0].index == res[1].index)
    assert np.all(res[0].values == res[1].values)


def test_multiple_outputs_vectorized_features(dummy_data):
    def sum_mean(x, axis):
        s = np.sum(x, axis)
        return s, s / x.shape[axis]

    fs = 4  # The sample frequency in Hz
    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(np.sum, "EDA", 250 * fs, 75 * fs),
            FeatureDescriptor(np.mean, "EDA", 250 * fs, 75 * fs),
            FeatureDescriptor(
                FuncWrapper(
                    sum_mean, output_names=["sum_vect", "mean_vect"],
                    vectorized=True, axis=1
                ),
                "EDA", 250 * fs, 75 * fs,
            )
        ]
    )

    res = fc.calculate(dummy_data, return_df=True)

    assert res.shape[1] == 4
    s = "EDA__"; p = "__w=1000"
    assert np.all(res[s+"sum"+p].values == res[s+"sum_vect"+p].values)
    assert np.all(res[s+"mean"+p].values == res[s+"mean_vect"+p].values)


def test_multiple_inputs_vectorized_features(dummy_data):
    def windowed_diff(x1, x2):
        return np.sum(x1, axis=-1) - np.sum(x2, axis=-1)

    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(np.sum, "EDA", "5min", "2.5min"),
            FeatureDescriptor(np.sum, "TMP", "5min", "2.5min"),
            FeatureDescriptor(
                FuncWrapper(windowed_diff, vectorized=True),
                ("EDA", "TMP"), "5min", "2.5min"
            )
        ]
    )

    res = fc.calculate(dummy_data, return_df=True)

    assert res.shape[1] == 3
    assert res.shape[0] > 1
    p = "__w=5m"
    manual_diff = res["EDA__sum"+p].values - res["TMP__sum"+p].values
    assert np.all(res["EDA|TMP__windowed_diff"+p].values == manual_diff)


### Test feature extraction length


def test_feature_extraction_length_range_index():
    s = pd.Series([0, 1, 2, 3, 4, 5], name="dummy")

    ## Case 1: stride = 1 sample
    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(np.min, "dummy", window=3, stride=1),
            FeatureDescriptor(
                FuncWrapper(np.min, output_names="min_", vectorized=True, axis=-1),
                "dummy", window=3, stride=1,
            )
        ]
    )

    res: List[pd.DataFrame]
    res = fc.calculate(s, window_idx="begin")
    assert len(res) == 2
    assert np.all(res[0].index == res[1].index)
    assert np.all(res[0].values == res[1].values)
    assert np.all(res[0].index.values == [0, 1, 2])
    assert np.all(res[0].values.ravel() == [0, 1, 2])

    # -> include final window [3, 4, 5]
    res = fc.calculate(s, window_idx="begin", include_final_window=True)
    assert len(res) == 2
    assert np.all(res[0].index == res[1].index)
    assert np.all(res[0].values == res[1].values)
    assert np.all(res[0].index.values == [0, 1, 2, 3])
    assert np.all(res[0].values.ravel() == [0, 1, 2, 3])

    ##  Case 2: stride = 3 (i.e., tumbling window\)
    # note: no vectorized featuredescriptor as vectorized functions require all 
    # segmented windows to have equal length
    fc = FeatureCollection(FeatureDescriptor(np.min, "dummy", window=3, stride=3))

    res = fc.calculate(s, window_idx="begin")
    assert len(res) == 1
    assert res[0].index == [0]
    assert res[0].values == [0]

    # -> extract on partially empty window (hence no vectorized)
    res = fc.calculate(s, window_idx="begin", include_final_window=True,
                       approve_sparsity=True)
    assert len(res) == 1
    assert np.all(res[0].index == [0, 3])
    assert np.all(res[0].values.ravel() == [0, 3])

    ## Case 3: some more additional testing for tumbling windows
    s = pd.Series(np.arange(10), name="dummy")
    assert len(s) == 10

    fc = FeatureCollection(
        feature_descriptors=[
            FeatureDescriptor(np.max, "dummy", window=2, stride=2),
            FeatureDescriptor(
                FuncWrapper(np.max, output_names="max_", vectorized=True, axis=-1),
                "dummy", window=2, stride=2,
            )
        ]
    )

    res = fc.calculate(s, window_idx="begin")
    assert len(res) == 2
    assert (len(res[0]) == 4) and (len(res[1]) == 4)
    assert np.all(res[0].index == res[1].index)
    assert np.all(res[0].values == res[1].values)
    assert np.all(res[0].index == [0, 2, 4, 6])
    for c in [0, 1]:
        assert np.all(res[c].values.ravel() == [1, 3, 5, 7])

    res = fc.calculate(s, window_idx="begin", include_final_window=True)
    assert len(res) == 2
    assert (len(res[0]) == 5) and (len(res[1]) == 5)
    assert np.all(res[0].index == res[1].index)
    assert np.all(res[0].values == res[1].values)
    assert np.all(res[0].index == [0, 2, 4, 6, 8])
    for c in [0, 1]:
        assert np.all(res[c].values.ravel() == [1, 3, 5, 7, 9])


def test_feature_extraction_length_float_index():
    s = pd.Series([0, 1, 2, 3, 4, 5], name="dummy")
    s.index = [0, 1, 2, 2.5, 3, 4]

    fc = FeatureCollection(FeatureDescriptor(np.min, "dummy", 3, 1))

    res = fc.calculate(s, window_idx="begin", approve_sparsity=True)
    assert len(res) == 1
    assert np.all(res[0].index.values == [0, 1])
    assert np.all(res[0].values.ravel() == [0, 1])

    res = fc.calculate(s, window_idx="begin", include_final_window=True,
                       approve_sparsity=True)
    assert len(res) == 1
    assert np.all(res[0].index.values == [0, 1, 2])
    assert np.all(res[0].values.ravel() == [0, 1, 2])


def test_feature_extraction_length_time_index():
    s = pd.Series([0, 1, 2, 3, 4, 5], name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=6)
    s.index = time_index

    fc = FeatureCollection(FeatureDescriptor(np.min, "dummy", "3h", "1h"))

    res = fc.calculate(s, window_idx="begin")
    assert len(res) == 1
    assert np.all(res[0].index.values == time_index[:3])
    assert np.all(res[0].values.ravel() == [0, 1, 2])

    res = fc.calculate(s, window_idx="begin", include_final_window=True)
    assert len(res) == 1
    assert np.all(res[0].index.values == time_index[:4])
    assert np.all(res[0].values.ravel() == [0, 1, 2, 3])


### Test 'error' use-cases


def test_type_error_add_feature_collection(dummy_data):
    fd = FeatureDescriptor(
        function=np.sum,
        series_name="EDA",
        window="5s",
        stride="2.5s",
    )
    fc = FeatureCollection(feature_descriptors=fd)

    with pytest.raises(TypeError):
        fc.add(np.sum)


def test_error_add_feature_collection_same_func_window(dummy_data):
    fd = FeatureDescriptor(np.sum, "EDA", window="5s", stride="2.5s")
    fc = FeatureCollection(feature_descriptors=fd)
    fd2 = FeatureDescriptor(np.sum, "EDA", window="5s", stride="8s")

    with pytest.raises(Exception):
        fc.add(fd2)


def test_one_to_many_error_feature_collection(dummy_data):
    def quantiles(sig: pd.Series) -> Tuple[float, float, float]:
        return np.quantile(sig, q=[0.1, 0.5, 0.9])

    # quantiles should be wrapped in a FuncWrapper
    fd = FeatureDescriptor(quantiles, series_name="EDA", window="5s", stride="2.5s")
    fc = FeatureCollection(fd)

    with pytest.raises(Exception):
        fc.calculate(dummy_data)


def test_one_to_many_wrong_np_funcwrapper_error_feature_collection(dummy_data):
    def quantiles(sig: pd.Series) -> Tuple[float, float, float]:
        return np.quantile(sig, q=[0.1, 0.5, 0.9])

    # Wrong number of output_names in func wrapper
    q_func = FuncWrapper(quantiles, output_names=["q_0.1", "q_0.5"])
    fd = FeatureDescriptor(q_func, series_name="EDA", window="5s", stride="2.5s")
    fc = FeatureCollection(fd)

    with pytest.raises(Exception):
        fc.calculate(dummy_data)


def test_many_to_one_error_feature_collection(dummy_data):
    def abs_mean_diff(sig1: pd.Series, sig2: pd.Series) -> float:
        # Note that this func only works when sig1 and sig2 have the same length
        return np.mean(np.abs(sig1 - sig2))

    # Give wrong nb of series names in tuple
    fd = FeatureDescriptor(
        abs_mean_diff, series_name=("EDA", "TMP", "ACC_x"), window="5s", stride="2.5s"
    )
    fc = FeatureCollection(fd)

    with pytest.raises(Exception):
        fc.calculate(dummy_data)


def test_error_same_output_feature_collection(dummy_data):
    def sum_func(sig: np.ndarray) -> float:
        return sum(sig)

    mfd = MultipleFeatureDescriptors(
        functions=[sum_func, FuncWrapper(np.max), np.min],
        series_names=["EDA", "TMP"],
        windows=["5s", "7s", "5s"],  # Two times 5s
        strides="2.5s",
    )
    with pytest.raises(AssertionError):
        fc = FeatureCollection(feature_descriptors=mfd)


def test_bound_method(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=[
            MultipleFeatureDescriptors(
                windows=[480, 1000],
                strides=480,
                functions=[np.mean, np.min, np.max, np.std],
                series_names=["TMP", "EDA"],
            )
        ]
    )

    df_tmp = dummy_data["TMP"].reset_index(drop=True)
    df_eda = dummy_data["EDA"].reset_index(drop=True).astype(float)
    df_tmp.index += 2

    for bound_method in ["inner", "outer", "inner-outer"]:
        fc.calculate(
            [df_tmp, df_eda],
            window_idx="middle",
            return_df=True,
            approve_sparsity=True,
            bound_method=bound_method,
        )

    with pytest.raises(ValueError):
        fc.calculate(
            [df_tmp, df_eda],
            window_idx="end",
            return_df=True,
            bound_method="invalid name",
        )


def test_bound_method_uneven_index_numeric(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=[
            MultipleFeatureDescriptors(
                windows=1000,
                strides=500,
                functions=[np.min, np.max],
                series_names=["TMP", "EDA"],
            )
        ]
    )

    df_tmp_ = dummy_data["TMP"].reset_index(drop=True)
    df_eda_ = dummy_data["EDA"].reset_index(drop=True)
    df_eda_.index = df_eda_.index.astype(float)
    df_eda_.index += 2.33

    latest_start = df_eda_.index[0]
    earliest_start = df_tmp_.index[0]

    out_inner = fc.calculate(
        [df_tmp_, df_eda_], bound_method="inner", window_idx="begin", return_df=True
    )
    assert out_inner.index[0] == latest_start

    out_outer = fc.calculate(
        [df_tmp_, df_eda_], bound_method="outer", window_idx="begin", return_df=True
    )
    assert out_outer.index[0] == earliest_start


def test_bound_method_uneven_index_datetime(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=[
            MultipleFeatureDescriptors(
                windows="5min",
                strides="3min",
                functions=[np.min, np.max],
                series_names=["TMP", "EDA"],
            )
        ]
    )

    df_tmp = dummy_data["TMP"]
    df_eda = dummy_data["EDA"]
    df_eda.index += pd.Timedelta(seconds=10)

    latest_start = df_eda.index[0]
    earliest_start = df_tmp.index[0]

    out_inner = fc.calculate(
        [df_tmp, df_eda], bound_method="inner", window_idx="begin", return_df=True
    )
    assert out_inner.index[0] == latest_start

    out_outer = fc.calculate(
        [df_tmp, df_eda], bound_method="outer", window_idx="begin", return_df=True
    )
    assert out_outer.index[0] == earliest_start


def test_bound_method_uneven_index_datetime_sequence(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=[
            MultipleFeatureDescriptors(
                windows=300,  # Sample based -> TimeIndexSampleStridedRolling
                strides=180,
                functions=[np.min, np.max],
                series_names=["TMP", "EDA"],
            )
        ]
    )

    df_tmp = dummy_data["TMP"]
    df_eda = dummy_data["EDA"]
    df_eda.index += pd.Timedelta(seconds=10)

    latest_start = df_eda.index[0]
    earliest_start = df_tmp.index[0]

    out_inner = fc.calculate(
        [df_tmp, df_eda], bound_method="inner", window_idx="begin", return_df=True
    )
    assert out_inner.index[0] == latest_start

    out_outer = fc.calculate(
        [df_tmp, df_eda], bound_method="outer", window_idx="begin", return_df=True
    )
    assert out_outer.index[0] == earliest_start


def test_not_sorted_fc(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=[
            MultipleFeatureDescriptors(
                windows=[480, 1000],
                strides=480,
                functions=[np.min, np.max],
                series_names=["TMP", "EDA"],
            )
        ]
    )

    df_tmp = dummy_data["TMP"].reset_index(drop=True)
    df_eda = dummy_data["EDA"].reset_index(drop=True).sample(frac=1)
    assert not df_eda.index.is_monotonic_increasing
    out = fc.calculate([df_tmp, df_eda], window_idx="end", return_df=True)
    assert not df_eda.index.is_monotonic_increasing

    df_eda.index = df_eda.index.astype(float)
    assert not df_eda.index.is_monotonic_increasing
    out = fc.calculate([df_tmp, df_eda], window_idx="end", return_df=True)
    assert not df_eda.index.is_monotonic_increasing


def test_serialization(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=[
            MultipleFeatureDescriptors(
                windows=[480, 1000],
                strides=480,
                functions=[np.mean, np.min, np.max, np.std],
                series_names=["TMP", "EDA"],
            )
        ]
    )

    df_tmp = dummy_data["TMP"].reset_index(drop=True)
    df_eda = dummy_data["EDA"].reset_index(drop=True)
    out = fc.calculate([df_tmp, df_eda], return_df=True)
    col_order = out.columns

    save_path = Path("featurecollection.pkl")
    if save_path.exists():
        os.remove(save_path)
    assert not save_path.exists()
    fc.serialize(save_path)
    assert save_path.exists() and save_path.is_file()

    fc_deserialized = dill.load(open(save_path, "rb"))
    out_deserialized = fc_deserialized.calculate([df_tmp, df_eda], return_df=True)
    assert np.allclose(
        out[col_order].values, out_deserialized[col_order].values, equal_nan=True
    )
    os.remove(save_path)


def test_vectorized_irregularly_sampled_data(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=FeatureDescriptor(
            FuncWrapper(np.std, vectorized=True, axis=1),
            "EDA", window="5min", stride="3s"
        )
    )

    df_eda = dummy_data["EDA"].dropna()
    df_eda.iloc[[3 + 66 * i for i in range(5)]] = np.nan
    df_eda = df_eda.dropna()

    assert len(df_eda) < len(dummy_data["EDA"].dropna())

    # Fails bc of irregularly sampled data
    # -> is a strict requirement to apply a vectorized feature function
    with pytest.raises(Exception):
        fc.calculate(df_eda)


def test_vectorized_multiple_asynchronous_strides(dummy_data):
    fc = FeatureCollection(
        feature_descriptors=FeatureDescriptor(
            FuncWrapper(np.std, vectorized=True, axis=1),
            "EDA", window="5min", stride=["3s", "5s"]
        )
    )

    df_eda = dummy_data["EDA"].dropna()

    # Fails bc of multiple asynchronous strides (resulting in different step sizes between the windows)
    # -> is a strict requirement to apply a vectorized feature function
    with pytest.raises(Exception):
        fc.calculate(df_eda)


def test_error_pass_stride_and_setpoints_calculate(dummy_data):
    setpoints = [0, 5, 7, 10]
    setpoints = dummy_data.index[setpoints].values
    fc = FeatureCollection(
        FeatureDescriptor(np.min, "EDA", window="5min", stride="3min")
    )

    # Fails because both a stride and setpoints is passed to the calculate method
    with pytest.raises(Exception):
        fc.calculate(dummy_data["EDA"], stride="3min", setpoints=setpoints)


def test_error_no_stride_and_no_setpoints(dummy_data):
    fc = FeatureCollection(
        [
            FeatureDescriptor(np.min, "EDA", window="5s"),
            FeatureDescriptor(np.max, "EDA", window="5s", stride="5s")
        ]
    )

    # Fails because at least one feature descriptor in the feature collection does not
    # have a stride, and no stride nor setpoints is passed to the calculate method
    with pytest.raises(Exception):
        fc.calculate(dummy_data)


def test_feature_collection_various_timezones_data():
    s_usa = pd.Series(
        [0, 1, 2, 3, 4, 5], 
        index=pd.date_range("2020-01-01", freq="1h", periods=6, tz='America/Chicago'),
        name="s_usa"
    )
    s_eu = pd.Series(
        [0, 1, 2, 3, 4, 5], 
        index=pd.date_range("2020-01-01", freq="1h", periods=6, tz='Europe/Brussels'),
        name="s_eu"
    )
    s_none = pd.Series(
        [0, 1, 2, 3, 4, 5], 
        index=pd.date_range("2020-01-01", freq="1h", periods=6, tz=None),
        name="s_none"
    )

    # As long as all features are calculated on the same tz data no error should be thrown
    for name in ["s_usa", "s_eu", "s_none"]:
        fc = FeatureCollection(
            MultipleFeatureDescriptors(np.min, name, "3h", "3h")
        )
        fc.calculate([s_usa, s_eu, s_none])

    # When feature collection calculates (different) features on different tz data
    # -> comparison error will be thrown
    fc = FeatureCollection(
        MultipleFeatureDescriptors(np.min, ["s_usa", "s_eu", "s_none"], "3h", "3h")
    )
    with pytest.raises(Exception):
        fc.calculate([s_usa, s_eu, s_none])
    fc = FeatureCollection(
        MultipleFeatureDescriptors(np.min, ["s_usa", "s_eu"], "3h", "3h")
    )
    with pytest.raises(Exception):
        fc.calculate([s_usa, s_eu, s_none])
    fc = FeatureCollection(
        MultipleFeatureDescriptors(np.min, ["s_usa", "s_none"], "3h", "3h")
    )
    with pytest.raises(Exception):
        fc.calculate([s_usa, s_eu, s_none])


def test_feature_collection_various_timezones_setpoints():
    # TODO: do we really want to support setpoints with different time zones?
    s_usa = pd.Series(
        [0, 1, 2, 3, 4, 5], 
        index=pd.date_range("2020-01-01", freq="1h", periods=6, tz='America/Chicago'),
        name="s_usa"
    )
    s_eu = pd.Series(
        [0, 1, 2, 3, 4, 5], 
        index=pd.date_range("2020-01-01", freq="1h", periods=6, tz='Europe/Brussels'),
        name="s_eu"
    )
    s_none = pd.Series(
        [0, 1, 2, 3, 4, 5], 
        index=pd.date_range("2020-01-01", freq="1h", periods=6, tz=None),
        name="s_none"
    )

    # As long as all features are calculated on the same tz data no error should be thrown
    for s in [s_usa, s_eu, s_none]:
        fc = FeatureCollection(
            FeatureDescriptor(np.min, s.name, "3h", "3h")
        )
        res = fc.calculate([s_usa, s_eu, s_none], setpoints=s.index[:3], n_jobs=0, return_df=True)
        assert np.all(res.values.ravel() == [0, 1, 2])

    fc = FeatureCollection(
        FeatureDescriptor(np.min, "s_usa", "3h", "3h")
    )
    res = fc.calculate(s_usa, setpoints=s_eu.index[:3].values, n_jobs=0, return_df=True)
    assert np.all(res.values == [])

    fc = FeatureCollection(
        FeatureDescriptor(np.min, "s_usa", "3h", "3h")
    )
    res = fc.calculate(s_usa, setpoints=s_none.index[:3].values, n_jobs=0, return_df=True)
    assert np.all(res.values == [])
