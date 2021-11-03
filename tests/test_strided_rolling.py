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
            stride=pd.Timedelta(seconds=2.5),
            window_idx=window_idx,
        )

        f = FuncWrapper(np.mean, output_names="numpy_mean")
        out = stroll.apply_func(f)
        assert f"EDA__numpy_mean__w=5s_s=2.5s" in out.columns

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


def test_abstract_class(dummy_data):
    tmp_series = dummy_data["TMP"].reset_index(drop=True)

    with pytest.raises(TypeError):
        StridedRolling(data=tmp_series, window=400, stride=100)