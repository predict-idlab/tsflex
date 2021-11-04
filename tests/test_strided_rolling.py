"""Tests for the strided rolling class"""

import numpy as np
import pandas as pd
import pytest

from .utils import dummy_data
from tsflex.features.segmenter.strided_rolling import (
    TimeStridedRolling,
    SequenceStridedRolling,
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
            data=tmp_series, window=4 * 30, stride=4 * 10, window_idx=window_idx,
            approve_sparsity=True,
        )

        f = FuncWrapper(np.mean, output_names="numpy_mean")
        out = stroll.apply_func(f)
        assert f"TMP__numpy_mean__w={int(4*30)}_s={int(4*10)}" in out.columns


def test_sequence_stroll_last_window_full(dummy_data):
    df_eda = dummy_data['EDA'].reset_index(drop=True)

    def stroll_apply_dummy_func(data, window, stride) -> pd.DataFrame:
        stroll = SequenceStridedRolling(data, window, stride, window_idx='end')
        return stroll.apply_func(FuncWrapper(np.min))

    out = stroll_apply_dummy_func(df_eda[:2201], window=1000, stride=200)
    assert out.index[-1] == 2200

    out = stroll_apply_dummy_func(df_eda[:2399], window=1000, stride=200)
    assert out.index[-1] == 2200

    # -> slicing is include left bound, discard right bound -> so UNTIL index 2200
    # i.e. last index in sequence is 2199 -> last valid full range 2200
    out = stroll_apply_dummy_func(df_eda[:2400], window=1000, stride=200)
    assert out.index[-1] == 2200
    out = stroll_apply_dummy_func(df_eda[:2401], window=1000, stride=200)
    assert out.index[-1] == 2400
    out = stroll_apply_dummy_func(df_eda[:2530], window=1000, stride=200)
    assert out.index[-1] == 2400


def test_time_stroll_last_window_full(dummy_data):
    df_eda = dummy_data['EDA']
    fs = 4

    def stroll_apply_dummy_func(data, window, stride) -> pd.DataFrame:
        stroll = TimeStridedRolling(data, window, stride, window_idx='end')
        return stroll.apply_func(FuncWrapper(np.min))

    window_s = pd.Timedelta('30s')
    stride_s = pd.Timedelta('10s')
    out = stroll_apply_dummy_func(df_eda[:fs*401], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs * 400]

    out = stroll_apply_dummy_func(df_eda[:fs*409], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs * 400]

    # -> slicing is include left bound, discard right bound -> so UNTIL index 410
    # i.e. last index in sequence is just without 410 -> last valid full range 400
    out = stroll_apply_dummy_func(df_eda[:fs*410], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs * 400]

    out = stroll_apply_dummy_func(df_eda[:fs*411], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs*410]
    out = stroll_apply_dummy_func(df_eda[:fs*526], window=window_s, stride=stride_s)
    assert out.index[-1] == df_eda.index[fs*520]


def test_abstract_class(dummy_data):
    tmp_series = dummy_data["TMP"].reset_index(drop=True)

    with pytest.raises(TypeError):
        StridedRolling(data=tmp_series, window=400, stride=100)
