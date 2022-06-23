"""Tests for the strided rolling class"""

import numpy as np
import pandas as pd
import pytest

from .utils import dummy_data
from tsflex.features.segmenter.strided_rolling import (
    TimeStridedRolling,
    SequenceStridedRolling,
    TimeIndexSampleStridedRolling,
    StridedRolling,
)
from tsflex.features import FuncWrapper


def test_time_stroll_window_idx(dummy_data):
    for window_idx in ["begin", "middle", "end"]:
        stroll = TimeStridedRolling(
            data=dummy_data[["EDA"]],
            window=pd.Timedelta(seconds=5),
            stride=pd.Timedelta(seconds=5),
            window_idx=window_idx,
        )

        f = FuncWrapper(np.mean, output_names="numpy_mean")
        out = stroll.apply_func(f)
        assert f"EDA__numpy_mean__w=5s_s=5s" in out.columns

    with pytest.raises(ValueError):
        TimeStridedRolling(
            data=dummy_data[["EDA", "TMP"]],
            window=pd.Timedelta(seconds=5),
            stride=pd.Timedelta(seconds=2.5),
            window_idx="invalid name",
        )


def test_sequence_stroll(dummy_data):
    tmp_series = dummy_data["TMP"].reset_index(drop=True)
    tmp_series[10:20] = None
    tmp_series = tmp_series.dropna()

    for window_idx in ["begin", "middle", "end"]:
        stroll = SequenceStridedRolling(
            data=tmp_series,
            window=4 * 30,
            stride=4 * 10,
            window_idx=window_idx,
            approve_sparsity=True,
        )

        f = FuncWrapper(np.mean, output_names="numpy_mean")
        out = stroll.apply_func(f)
        assert f"TMP__numpy_mean__w={int(4*30)}_s={int(4*10)}" in out.columns


def test_sequence_stroll_last_window_full(dummy_data):
    df_eda = dummy_data["EDA"].reset_index(drop=True)

    def stroll_apply_dummy_func(data, window, stride) -> pd.DataFrame:
        stroll = SequenceStridedRolling(data, window, stride, window_idx="end")
        return stroll.apply_func(FuncWrapper(np.min))

    out = stroll_apply_dummy_func(df_eda[:2198], window=1000, stride=200)
    assert out.index[-1] == 2000
    out = stroll_apply_dummy_func(df_eda[:2199], window=1000, stride=200)
    assert out.index[-1] == 2000
    out = stroll_apply_dummy_func(df_eda[:2200], window=1000, stride=200)
    assert out.index[-1] == 2000
    out = stroll_apply_dummy_func(df_eda[:2201], window=1000, stride=200)
    assert out.index[-1] == 2200
    out = stroll_apply_dummy_func(df_eda[:2202], window=1000, stride=200)
    assert out.index[-1] == 2200

    def stroll_apply_dummy_func(data, window, stride) -> pd.DataFrame:
        stroll = SequenceStridedRolling(data, window, stride, window_idx="begin")
        return stroll.apply_func(FuncWrapper(np.min))

    out = stroll_apply_dummy_func(df_eda[:2198], window=1000, stride=200)
    assert out.index[-1] == 1000
    out = stroll_apply_dummy_func(df_eda[:2199], window=1000, stride=200)
    assert out.index[-1] == 1000
    out = stroll_apply_dummy_func(df_eda[:2200], window=1000, stride=200)
    assert out.index[-1] == 1000
    out = stroll_apply_dummy_func(df_eda[:2201], window=1000, stride=200)
    assert out.index[-1] == 1200
    out = stroll_apply_dummy_func(df_eda[:2202], window=1000, stride=200)
    assert out.index[-1] == 1200


def test_time_stroll_last_window_full(dummy_data):
    df_eda = dummy_data["EDA"]
    fs = 4

    def stroll_apply_dummy_func(data, window, stride) -> pd.DataFrame:
        stroll = TimeStridedRolling(data, window, stride, window_idx="end")
        return stroll.apply_func(FuncWrapper(np.min))

    window_s = pd.Timedelta("30s")
    stride_s = pd.Timedelta("10s")
    out = stroll_apply_dummy_func(df_eda[: fs * 401], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs * 400]

    out = stroll_apply_dummy_func(df_eda[: fs * 409], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs * 400]

    # -> slicing is include left bound, discard right bound -> so UNTIL index 410
    # i.e. last index in sequence is just without 410 -> last valid full range 400
    out = stroll_apply_dummy_func(df_eda[: fs * 410], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs * 400]

    out = stroll_apply_dummy_func(df_eda[: fs * 411], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs * 410]
    out = stroll_apply_dummy_func(df_eda[: fs * 526], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs * 520]


def test_abstract_class(dummy_data):
    tmp_series = dummy_data["TMP"].reset_index(drop=True)

    with pytest.raises(TypeError):
        StridedRolling(data=tmp_series, window=400, stride=400)


def test_time_index_sequence_stroll(dummy_data):
    df_eda = dummy_data["EDA"]
    stroll = TimeIndexSampleStridedRolling(
        df_eda, window=1000, stride=50, window_idx="end"
    )
    return stroll.apply_func(FuncWrapper(np.min))


def test_sequence_stroll_indexing():
    s = pd.Series(data=[0,1,2,3,4], name="dummy")

    ## No Force
    sr = SequenceStridedRolling(s, window=3, stride=1, window_idx="begin")
    assert np.all(sr.index == [0,1])
    sr = SequenceStridedRolling(s, window=3, stride=2, window_idx="begin")
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, stride=3, window_idx="begin")
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, stride=4, window_idx="begin")
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, stride=5, window_idx="begin")
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, stride=50, window_idx="begin")
    assert np.all(sr.index == [0])

    sr = SequenceStridedRolling(s, window=4, stride=1, window_idx="begin")
    assert np.all(sr.index == [0])

    sr = SequenceStridedRolling(s, window=5, stride=1, window_idx="begin")
    assert np.all(sr.index == [])
    sr = SequenceStridedRolling(s, window=6, stride=1, window_idx="begin")
    assert np.all(sr.index == [])

    ## Force
    sr = SequenceStridedRolling(s, window=3, stride=1, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,1,2])
    sr = SequenceStridedRolling(s, window=3, stride=2, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,2])
    sr = SequenceStridedRolling(s, window=3, stride=4, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0, 4])
    sr = SequenceStridedRolling(s, window=3, stride=5, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, stride=50, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])

    sr = SequenceStridedRolling(s, window=4, stride=1, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0, 1])

    sr = SequenceStridedRolling(s, window=5, stride=1, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=6, stride=1, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [])


def test_time_stroll_indexing():
    s = pd.Series(data=[0,1,2,3,4], name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=5)
    s.index = time_index

    def get_time_index(arr):
        return [time_index[idx] for idx in arr]

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(1, unit="h"), window_idx="begin")
    assert np.all(sr.index == get_time_index([0,1]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(2, unit="h"), window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(3, unit="h"), window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(4, unit="h"), window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(5, unit="h"), window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(50, unit="h"), window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(4, unit="h"), stride=pd.Timedelta(1, unit="h"), window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), stride=pd.Timedelta(1, unit="h"), window_idx="begin")
    assert np.all(sr.index == [])
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), stride=pd.Timedelta(1, unit="h"), window_idx="begin")
    assert np.all(sr.index == [])

    ## Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(1, unit="h"), window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0,1,2]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(2, unit="h"), window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0,2]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(4, unit="h"), window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0, 4]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(5, unit="h"), window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), stride=pd.Timedelta(50, unit="h"), window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(4, unit="h"), stride=pd.Timedelta(1, unit="h"), window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0, 1]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), stride=pd.Timedelta(1, unit="h"), window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), stride=pd.Timedelta(1, unit="h"), window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [])
