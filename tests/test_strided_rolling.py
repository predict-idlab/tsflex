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
            strides=pd.Timedelta(seconds=5),
            window_idx=window_idx,
        )

        f = FuncWrapper(np.mean, output_names="numpy_mean")
        out = stroll.apply_func(f)
        assert "EDA__numpy_mean__w=5s" in out.columns

    with pytest.raises(ValueError):
        TimeStridedRolling(
            data=dummy_data[["EDA", "TMP"]],
            window=pd.Timedelta(seconds=5),
            strides=[pd.Timedelta(seconds=2.5)],
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
            strides=4 * 10,
            window_idx=window_idx,
            approve_sparsity=True,
        )

        f = FuncWrapper(np.mean, output_names="numpy_mean")
        out = stroll.apply_func(f)
        assert f"TMP__numpy_mean__w={int(4*30)}" in out.columns


def test_sequence_stroll_last_window_full(dummy_data):
    df_eda = dummy_data["EDA"].reset_index(drop=True)

    def stroll_apply_dummy_func(data, window, stride) -> pd.DataFrame:
        stroll = SequenceStridedRolling(data, window, [stride], window_idx="end")
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
        stroll = TimeStridedRolling(data, window, [stride], window_idx="end")
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
        StridedRolling(data=tmp_series, window=400, strides=[400])


def test_time_index_sequence_stroll(dummy_data):
    df_eda = dummy_data["EDA"]
    window, stride = 1000, 50
    stroll = TimeIndexSampleStridedRolling(
        df_eda, window=window, strides=[stride], window_idx="end"
    )
    out = stroll.apply_func(FuncWrapper(np.min))
    assert out.index[0] == df_eda.index[window]
    assert out.index[1] == df_eda.index[window + stride]
    assert out.index[2] == df_eda.index[window + 2 * stride]


def test_sequence_stroll_indexing():
    s = pd.Series(data=[0, 1, 2, 3, 4], name="dummy")

    ## No Force
    sr = SequenceStridedRolling(s, window=3, strides=[1], window_idx="begin")
    assert np.all(sr.index == [0,1])
    sr = SequenceStridedRolling(s, window=3, strides=[2], window_idx="begin")
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, strides=[3], window_idx="begin")
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, strides=[4], window_idx="begin")
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, strides=[5], window_idx="begin")
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, strides=[50], window_idx="begin")
    assert np.all(sr.index == [0])

    sr = SequenceStridedRolling(s, window=4, strides=[1], window_idx="begin")
    assert np.all(sr.index == [0])

    sr = SequenceStridedRolling(s, window=5, strides=[1], window_idx="begin")
    assert np.all(sr.index == [])
    sr = SequenceStridedRolling(s, window=5, strides=[2], window_idx="begin")
    assert np.all(sr.index == [])
    sr = SequenceStridedRolling(s, window=6, strides=[1], window_idx="begin")
    assert np.all(sr.index == [])
    sr = SequenceStridedRolling(s, window=6, strides=[2], window_idx="begin")
    assert np.all(sr.index == [])

    ## Force
    sr = SequenceStridedRolling(s, window=3, strides=[1], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,1,2])
    sr = SequenceStridedRolling(s, window=3, strides=[2], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,2])
    sr = SequenceStridedRolling(s, window=3, strides=[4], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0, 4])
    sr = SequenceStridedRolling(s, window=3, strides=[5], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=3, strides=[50], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])

    sr = SequenceStridedRolling(s, window=4, strides=[1], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0, 1])

    sr = SequenceStridedRolling(s, window=5, strides=[1], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=5, strides=[2], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=6, strides=[1], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = SequenceStridedRolling(s, window=6, strides=[2], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])


def test_time_stroll_indexing():
    s = pd.Series(data=[0, 1, 2, 3, 4], name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=5)
    s.index = time_index

    def get_time_index(arr):
        return [time_index[idx] for idx in arr]

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0,1]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(4, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(5, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(50, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(4, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert np.all(sr.index == [])
    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin")
    assert np.all(sr.index == [])
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert np.all(sr.index == [])
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin")
    assert np.all(sr.index == [])

    ## Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0,1,2]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0,2]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(4, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0, 4]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(5, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(50, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(4, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0, 1]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))


def test_time_index_sequence_stroll_indexing():
    # Same test as above, but with an time-index and sequence arguments
    s = pd.Series(data=[0, 1, 2, 3, 4], name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=5)
    s.index = time_index

    ## No Force
    # Note -> the current TimeStridedRolling implementation stitches the time index back 
    # together based on the sequence index.
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=[1], window_idx="begin")
    assert np.all(sr.index == [0,1])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=[2], window_idx="begin")
    assert np.all(sr.index == [0])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=[3], window_idx="begin")
    assert np.all(sr.index == [0])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=4, window_idx="begin")
    assert np.all(sr.index == [0])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=5, window_idx="begin")
    assert np.all(sr.index == [0])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=50, window_idx="begin")
    assert np.all(sr.index == [0])

    sr = TimeIndexSampleStridedRolling(s, window=4, strides=50, window_idx="begin")
    assert np.all(sr.index == [0])

    sr = TimeIndexSampleStridedRolling(s, window=5, strides=1, window_idx="begin")
    assert np.all(sr.index == [])
    sr = TimeIndexSampleStridedRolling(s, window=5, strides=2, window_idx="begin")
    assert np.all(sr.index == [])
    sr = TimeIndexSampleStridedRolling(s, window=6, strides=1, window_idx="begin")
    assert np.all(sr.index == [])
    sr = TimeIndexSampleStridedRolling(s, window=6, strides=2, window_idx="begin")
    assert np.all(sr.index == [])

    ## Force
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=1, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,1,2])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=2, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,2])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=3, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,3])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=4, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,4])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=5, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = TimeIndexSampleStridedRolling(s, window=3, strides=50, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])

    sr = TimeIndexSampleStridedRolling(s, window=4, strides=1, window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0,1])

    sr = TimeIndexSampleStridedRolling(s, window=5, strides=[1], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = TimeIndexSampleStridedRolling(s, window=5, strides=[2], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = TimeIndexSampleStridedRolling(s, window=6, strides=[1], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == [0])
    sr = TimeIndexSampleStridedRolling(s, window=6, strides=[2], window_idx="begin", include_final_window=True)


def test_time_stroll_indexing_multiple_strides():
    s = pd.Series(data=np.arange(20), name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=20)
    s.index = time_index

    def get_time_index(arr):
        return [time_index[idx] for idx in arr]

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0,3,5,6,9,10,12,15]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(19, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], window_idx="begin")
    assert np.all(sr.index == get_time_index([0]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(20, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], window_idx="begin")
    assert np.all(sr.index == [])
    sr = TimeStridedRolling(s, window=pd.Timedelta(21, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], window_idx="begin")
    assert np.all(sr.index == [])

    ## Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0,3,5,6,9,10,12,15,18]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(19, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0, 3, 5]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(20, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))
    sr = TimeStridedRolling(s, window=pd.Timedelta(21, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], window_idx="begin", include_final_window=True)
    assert np.all(sr.index == get_time_index([0]))


def test_sequence_stroll_indexing_segment_start_idxs():
    segment_start_idxs = np.array([0, 5, 7, 10])
    s = pd.Series(data=np.arange(20), name="dummy")

    ## No Force
    sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_start_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_start_idxs=segment_start_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_start_idxs=segment_start_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs + 3)

    ## Force
    sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_start_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_start_idxs=segment_start_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_start_idxs=segment_start_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs + 3)


def test_sequence_stroll_indexing_segment_end_idxs():
    segment_end_idxs = np.array([5, 7, 10])
    s = pd.Series(data=np.arange(20), name="dummy")

    ## No Force
    sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_end_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_end_idxs=segment_end_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_end_idxs=segment_end_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs - 3)

    ## Force
    sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_end_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_end_idxs=segment_end_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_end_idxs=segment_end_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs - 3)


def test_sequence_stroll_indexing_segment_start_idxs_outside_valid_range():
    segment_start_idxs = np.array([0, 5, 7, 10, 200])
    s = pd.Series(data=np.arange(20), name="dummy")

    ## No Force
    sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_start_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_start_idxs=segment_start_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_start_idxs=segment_start_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs + 3)

    ## Force
    sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_start_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_start_idxs=segment_start_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_start_idxs=segment_start_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs + 3)


def test_sequence_stroll_indexing_segment_end_idxs_outside_valid_range():
    segment_end_idxs = np.array([3, 5, 7, 10, 200])
    s = pd.Series(data=np.arange(20), name="dummy")

    ## No Force
    sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_end_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_end_idxs=segment_end_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_end_idxs=segment_end_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs - 3)

    ## Force
    sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_end_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_end_idxs=segment_end_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = SequenceStridedRolling(s, window=3, strides=[3, 5], segment_end_idxs=segment_end_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs - 3)


def test_time_stroll_indexing_segment_start_idxs():
    s = pd.Series(data=np.arange(20), name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=20)
    s.index = time_index

    segment_start_idxs = s.index[[0, 5, 7, 10]].values

    def get_time_index(arr):
        return [time_index[idx] for idx in arr]

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_start_idxs=segment_start_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == get_time_index([0, 5, 7, 10]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_start_idxs=segment_start_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == get_time_index([0, 5, 7, 10]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_start_idxs=segment_start_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == [t + pd.Timedelta(3, unit="h") for t in get_time_index([0, 5, 7, 10])])

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_start_idxs=segment_start_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == get_time_index([0, 5, 7, 10]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_start_idxs=segment_start_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == get_time_index([0, 5, 7, 10]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_start_idxs=segment_start_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == [t + pd.Timedelta(3, unit="h") for t in get_time_index([0, 5, 7, 10])])


def test_time_stroll_indexing_segment_end_idxs():
    s = pd.Series(data=np.arange(20), name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=20)
    s.index = time_index

    segment_end_idxs = s.index[[0, 5, 7, 10]].values

    def get_time_index(arr):
        return [time_index[idx] for idx in arr]

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_end_idxs=segment_end_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == get_time_index([0, 5, 7, 10]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_end_idxs=segment_end_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == get_time_index([0, 5, 7, 10]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_end_idxs=segment_end_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == [t - pd.Timedelta(3, unit="h") for t in get_time_index([0, 5, 7, 10])])

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_end_idxs=segment_end_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == get_time_index([0, 5, 7, 10]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_end_idxs=segment_end_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == get_time_index([0, 5, 7, 10]))

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_end_idxs=segment_end_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == [t - pd.Timedelta(3, unit="h") for t in get_time_index([0, 5, 7, 10])])


def test_time_stroll_indexing_segment_start_idxs_outside_valid_range():
    s = pd.Series(data=np.arange(20), name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=20)
    s.index = time_index

    segment_start_idxs = s.index[[0, 5, 7, 10]].values
    segment_start_idxs = np.append(segment_start_idxs, (s.index[[0]] + pd.Timedelta(200, unit="h")).values)

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_start_idxs=segment_start_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_start_idxs=segment_start_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_start_idxs=segment_start_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == [t + pd.Timedelta(3, unit="h") for t in segment_start_idxs])

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_start_idxs=segment_start_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_start_idxs=segment_start_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_start_idxs)

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_start_idxs=segment_start_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == [t + pd.Timedelta(3, unit="h") for t in segment_start_idxs])


def test_time_stroll_indexing_segment_end_idxs_outside_valid_range():
    s = pd.Series(data=np.arange(20), name="dummy")
    time_index = pd.date_range("2020-01-01", freq="1h", periods=20)
    s.index = time_index

    segment_end_idxs = s.index[[0, 5, 7, 10]].values
    segment_end_idxs = np.append(segment_end_idxs, (s.index[[0]] + pd.Timedelta(200, unit="h")).values)

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_end_idxs=segment_end_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_end_idxs=segment_end_idxs, window_idx="end")
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_end_idxs=segment_end_idxs, window_idx="begin")
    assert sr.strides is None
    assert np.all(sr.index == [t - pd.Timedelta(3, unit="h") for t in segment_end_idxs])

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_end_idxs=segment_end_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_end_idxs=segment_end_idxs, window_idx="end", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == segment_end_idxs)

    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h"), pd.Timedelta(5, unit="h")], segment_end_idxs=segment_end_idxs, window_idx="begin", include_final_window=True)
    assert sr.strides is None
    assert np.all(sr.index == [t - pd.Timedelta(3, unit="h") for t in segment_end_idxs])



def test_sequence_stroll_apply_func_vectorized():
    f = FuncWrapper(np.min, output_names="min")
    f_vect = FuncWrapper(np.min, output_names="min_vect", vectorized=True, axis=-1)

    s = pd.Series(data=[0, 1, 2, 3, 4], name="dummy")

    def assert_1col_df_equal(s1, s2):
        assert (s1.shape[1] == 1) & (s2.shape[1] == 1)
        assert np.all(s1.index == s2.index)
        assert np.all(s1.values.ravel() == s2.values.ravel())

    ## No Force
    sr = SequenceStridedRolling(s, window=3, strides=[1], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[2], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[3], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[4], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[5], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[50], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = SequenceStridedRolling(s, window=4, strides=[1], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = SequenceStridedRolling(s, window=5, strides=[1], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=6, strides=[1], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    ## Force
    sr = SequenceStridedRolling(s, window=3, strides=[1], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[2], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    with pytest.raises(Exception):
        # the vectorized function requires the same number of samples in each segmented window
        # this will result in window of length 3 and 1
        sr = SequenceStridedRolling(s, window=3, strides=[4], window_idx="begin", include_final_window=True)
        sr.apply_func(f_vect)
    sr = SequenceStridedRolling(s, window=3, strides=[5], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[50], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = SequenceStridedRolling(s, window=4, strides=[1], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = SequenceStridedRolling(s, window=5, strides=[1], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=6, strides=[1], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))


def test_time_stroll_apply_func_vectorized():
    f = FuncWrapper(np.min, output_names="min")
    f_vect = FuncWrapper(np.min, output_names="min_vect", vectorized=True, axis=-1)

    s = pd.Series(data=[0, 1, 2, 3, 4], name="dummy")
    s.index = pd.date_range("2020-01-01", freq="1h", periods=5)

    def assert_1col_df_equal(s1, s2):
        assert (s1.shape[1] == 1) & (s2.shape[1] == 1)
        assert np.all(s1.index == s2.index)
        assert np.all(s1.values.ravel() == s2.values.ravel())

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(4, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(5, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(50, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = TimeStridedRolling(s, window=pd.Timedelta(4, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    ## Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    with pytest.raises(Exception):
        # the vectorized function requires the same number of samples in each segmented window
        # this will result in window of length 3 and 1
        sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(4, unit="h")], window_idx="begin", include_final_window=True)
        sr.apply_func(f_vect)
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(5, unit="h")], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(50, unit="h")], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = TimeStridedRolling(s, window=pd.Timedelta(4, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))


def test_sequence_stroll_apply_func_vectorized_segment_idxs():
    f = FuncWrapper(np.min, output_names="min")
    f_vect = FuncWrapper(np.min, output_names="min_vect", vectorized=True, axis=-1)

    segment_idxs = np.array([3, 6, 9])
    s = pd.Series(data=np.arange(20), name="dummy")

    def assert_1col_df_equal(s1, s2):
        assert (s1.shape[1] == 1) & (s2.shape[1] == 1)
        assert np.all(s1.index == s2.index)
        assert np.all(s1.values.ravel() == s2.values.ravel())

    ### START IDXS

    ## No Force
    sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_idxs, window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    ## Force
    sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_idxs, window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    with pytest.raises(Exception):
        # Because of irregular stride step in the segment_start_idxs
        segment_start_idxs = np.array([0, 2, 3])
        sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_start_idxs, window_idx="begin")
        sr.apply_func(f_vect)

    ### END IDXS

    ## No Force
    sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_idxs, window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    ## Force
    sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_idxs, window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    with pytest.raises(Exception):
        # Because of irregular stride step in the segment_start_idxs
        segment_end_idxs = np.array([0, 2, 3])
        sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_end_idxs, window_idx="begin")
        sr.apply_func(f_vect)
    

def test_time_stroll_apply_func_vectorized_segment_idxs():
    f = FuncWrapper(np.min, output_names="min")
    f_vect = FuncWrapper(np.min, output_names="min_vect", vectorized=True, axis=-1)

    s = pd.Series(data=np.arange(20), name="dummy")
    s.index = pd.date_range("2020-01-01", freq="1h", periods=20)
    segment_idxs = s.index[[3, 6, 9]].values

    def assert_1col_df_equal(s1, s2):
        assert (s1.shape[1] == 1) & (s2.shape[1] == 1)
        assert np.all(s1.index == s2.index)
        assert np.all(s1.values.ravel() == s2.values.ravel())

    ### START IDXS

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_start_idxs=segment_idxs, window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
   
    ## Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_start_idxs=segment_idxs, window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    with pytest.raises(Exception):
        # Because of irregular stride step in the segment_start_idxs
        segment_start_idxs = s.index[[0, 2, 3]].values
        sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_start_idxs, window_idx="begin")
        sr.apply_func(f_vect)

    ### END IDXS
    
    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_end_idxs=segment_idxs, window_idx="begin")
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
   
    ## Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_end_idxs=segment_idxs, window_idx="begin", include_final_window=True)
    assert_1col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    with pytest.raises(Exception):
        # Because of irregular stride step in the segment_end_idxs
        segment_end_idxs = s.index[[0, 2, 3]].values
        sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_end_idxs, window_idx="begin")
        sr.apply_func(f_vect)


def test_sequence_stroll_apply_func_segment_idxs_empty():
    f = FuncWrapper(np.min, output_names="min")

    segment_idxs = np.array([])
    s = pd.Series(data=np.arange(20), name="dummy")

    ## Start idxs
    sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_idxs, window_idx="begin")
    assert np.all(sr.index == [])
    res = sr.apply_func(f)
    assert len(res) == 0  

    ## End idxs
    sr = SequenceStridedRolling(s, window=3, segment_end_idxs=segment_idxs, window_idx="begin")
    assert np.all(sr.index == [])
    res = sr.apply_func(f)
    assert len(res) == 0  

    ## Start and end idxs
    sr = SequenceStridedRolling(s, window=3, segment_start_idxs=segment_idxs, segment_end_idxs=segment_idxs, window_idx="begin")
    assert np.all(sr.index == [])
    res = sr.apply_func(f)
    assert len(res) == 0  


def test_time_stroll_apply_func_segment_idxs_empty():
    f = FuncWrapper(np.min, output_names="min")

    s = pd.Series(data=np.arange(20), name="dummy")
    s.index = pd.date_range("2020-01-01", freq="1h", periods=20)
    segment_idxs = np.array([])

    ## Start idxs
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_start_idxs=segment_idxs, window_idx="begin")
    assert np.all(sr.index == [])
    res = sr.apply_func(f)
    assert len(res) == 0  

    ## End idxs
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_end_idxs=segment_idxs, window_idx="begin")
    assert np.all(sr.index == [])
    res = sr.apply_func(f)
    assert len(res) == 0  

    ## Start and end idxs
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), segment_start_idxs=segment_idxs, segment_end_idxs=segment_idxs, window_idx="begin")
    assert np.all(sr.index == [])
    res = sr.apply_func(f)
    assert len(res) == 0  


def test_sequence_stroll_apply_func_vectorized_multi_output():
    def min_max(arr, axis=None):
        return np.min(arr, axis=axis), np.max(arr, axis=axis)

    f = FuncWrapper(min_max, output_names=["min", "max"])
    f_vect = FuncWrapper(min_max, output_names=["min_vect", "max_vect"],
                         vectorized=True, axis=-1)

    s = pd.Series(data=[0, 1, 2, 3, 4], name="dummy")

    def assert_2col_df_equal(s1, s2):
        assert (s1.shape[1] == 2) & (s2.shape[1] == 2)
        assert np.all(s1.index == s2.index)
        for name in ["min", "max"]:
            assert np.all(s1.filter(like=name).values.ravel() == s2.filter(
                like=name).values.ravel())

    ## No Force
    sr = SequenceStridedRolling(s, window=3, strides=[1], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[2], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[3], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[4], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[5], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[50], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = SequenceStridedRolling(s, window=4, strides=[1], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = SequenceStridedRolling(s, window=5, strides=[1], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=6, strides=[1], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    ## Force
    sr = SequenceStridedRolling(s, window=3, strides=[1], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[2], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    with pytest.raises(Exception):
        # the vectorized function requires the same number of samples in each segmented window
        # this will result in window of length 3 and 1
        sr = SequenceStridedRolling(s, window=3, strides=[4], window_idx="begin", include_final_window=True)
        sr.apply_func(f_vect)
    sr = SequenceStridedRolling(s, window=3, strides=[5], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=3, strides=[50], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = SequenceStridedRolling(s, window=4, strides=[1], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = SequenceStridedRolling(s, window=5, strides=[1], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = SequenceStridedRolling(s, window=6, strides=[1], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))


def test_time_stroll_apply_func_vectorized_multi_output():
    def min_max(arr, axis=None):
        return np.min(arr, axis=axis), np.max(arr, axis=axis)

    f = FuncWrapper(min_max, output_names=["min", "max"])
    f_vect = FuncWrapper(min_max, output_names=["min_vect", "max_vect"],
                         vectorized=True, axis=-1)

    s = pd.Series(data=[0, 1, 2, 3, 4], name="dummy")
    s.index = pd.date_range("2020-01-01", freq="1h", periods=5)

    def assert_2col_df_equal(s1, s2):
        assert (s1.shape[1] == 2) & (s2.shape[1] == 2)
        assert np.all(s1.index == s2.index)
        for name in ["min", "max"]:
            assert np.all(s1.filter(like=name).values.ravel() == s2.filter(
                like=name).values.ravel())

    ## No Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(4, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(5, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(50, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = TimeStridedRolling(s, window=pd.Timedelta(4, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    ## Force
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(2, unit="h")], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    with pytest.raises(Exception):
        # the vectorized function requires the same number of samples in each segmented window
        # this will result in window of length 3 and 1
        sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(4, unit="h")], window_idx="begin", include_final_window=True)
        sr.apply_func(f_vect)
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(5, unit="h")], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(50, unit="h")], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = TimeStridedRolling(s, window=pd.Timedelta(4, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))

    sr = TimeStridedRolling(s, window=pd.Timedelta(5, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))
    sr = TimeStridedRolling(s, window=pd.Timedelta(6, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin", include_final_window=True)
    assert_2col_df_equal(sr.apply_func(f), sr.apply_func(f_vect))


def test_index_data_type_retention():
    s = pd.Series([0, 1, 2, 3, 4], name="dummy")

    ### Int
    sr = SequenceStridedRolling(s, window=3, strides=[1], window_idx="begin")
    assert str(sr.index.dtype).startswith("int")

    ### Float
    s.index = [0., 1., 2., 3., 4.]
    sr = SequenceStridedRolling(s, window=3, strides=[1], window_idx="begin")
    assert str(sr.index.dtype).startswith("float")

    ### Time
    s.index = pd.date_range("2020-01-01", freq="1h", periods=5)
    sr = TimeStridedRolling(s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(1, unit="h")], window_idx="begin")
    assert "datetime64" in str(sr.index.dtype)
    assert not str(sr.index.dtype).startswith("int")
    assert not str(sr.index.dtype).startswith("float")


def test_various_time_zones():
    f = FuncWrapper(np.min)
    f2 = FuncWrapper(np.dot)
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
        sr = TimeStridedRolling(
            s, window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h")]
        )
        res = sr.apply_func(f)
        assert np.all(res.values == [0])
        sr = TimeStridedRolling(
            [s, s], window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h")]
        )
        res = sr.apply_func(f2)
        assert np.all(res.values == [5])

    # When features are calculated (different) features on different tz data
    # -> error will be thrown
    with pytest.raises(Exception):
        sr = TimeStridedRolling(
            [s_usa, s_eu], window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h")]
        )
    with pytest.raises(Exception):
        sr = TimeStridedRolling(
            [s_usa, s_none], window=pd.Timedelta(3, unit="h"), strides=[pd.Timedelta(3, unit="h")]
        )
