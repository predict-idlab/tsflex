# -*- coding: utf-8 -*-
"""
"""
__author__ = 'Jonas Van Der Donckt'

from pandas.core.indexes.datetimes import DatetimeIndex
import pytest

from tsflex.features.segmenter import StridedRollingFactory
from tsflex.features import FuncWrapper
from tsflex.features.segmenter.strided_rolling import TimeIndexSequenceStridedRolling
from tsflex.utils.time import parse_time_arg

from .utils import dummy_data
import numpy as np
import pandas as pd


def test_stroll_time_data(dummy_data):
    def corr(s1, s2):
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        return np.corrcoef(s1, s2)[0][-1].astype(s1.dtype)

    f_corr = FuncWrapper(func=corr, output_names="corrcoef")
    window = parse_time_arg('30s')
    stride = parse_time_arg('30s')

    stroll = StridedRollingFactory.get_segmenter(
        data=dummy_data[['EDA', 'TMP']],
        window=window,
        stride=stride,
    )

    out_f = stroll.apply_func(f_corr)


def test_stroll_sequence_data(dummy_data):
    def corr(s1, s2):
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        return np.corrcoef(s1, s2)[0][-1].astype(s1.dtype)

    f_corr = FuncWrapper(func=corr, output_names="corrcoef")
    window = 30 * 4
    stride = 10 * 4

    eda_data = dummy_data['EDA'].reset_index(drop=True)
    tmp_data = dummy_data['TMP'].reset_index(drop=True)
    stroll = StridedRollingFactory.get_segmenter(
        data=[eda_data, tmp_data],
        window=window,
        stride=stride,
    )
    out_f = stroll.apply_func(f_corr)
    assert out_f.columns[0] == f'EDA|TMP__corrcoef__w={int(window)}_s={int(stride)}'


def test_stroll_sequence_int_float(dummy_data):
    def corr(s1, s2):
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        return np.corrcoef(s1, s2)[0][-1].astype(s1.dtype)

    f_corr = FuncWrapper(func=corr, output_names="corrcoef")
    window = 30 * 4
    stride = 10 * 4

    eda_data = dummy_data['EDA'].reset_index(drop=True)
    tmp_data = dummy_data['TMP'].reset_index(drop=True)

    eda_data.index = eda_data.index.astype('float')
    stroll = StridedRollingFactory.get_segmenter(
        data=[eda_data, tmp_data],
        window=window,
        stride=stride,
    )
    out_f = stroll.apply_func(f_corr)
    assert out_f.columns[0] == f'EDA|TMP__corrcoef__w={int(window)}_s={int(stride)}'


def test_stroll_mixed_index_dtypes(dummy_data):
    window = 30 * 4
    stride = 10 * 4

    # Note: we donnot reset the index of the EDA data to int-index
    eda_data = dummy_data['EDA'].reset_index(drop=True)
    tmp_data = dummy_data['TMP']

    with pytest.raises(ValueError):
        stroll = StridedRollingFactory.get_segmenter(
            data=[eda_data, tmp_data],
            window=window,
            stride=stride,
        )

def test_stroll_timeindex_sequence(dummy_data):
    # Note: we donnot reset the index of the EDA data to int-index
    eda_data = dummy_data['EDA']
    tmp_data = dummy_data['TMP']

    # we should get a TimeIndexSequenceStridedRolling segmenter
    window = 3000
    stride = 1000
    stroll = StridedRollingFactory.get_segmenter(
        data=[eda_data, tmp_data],
        window=window,
        stride=stride
    )

    assert isinstance(stroll, TimeIndexSequenceStridedRolling)

    def corr(s1, s2):
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        return np.corrcoef(s1, s2)[0][-1].astype(s1.dtype)
    f_corr = FuncWrapper(func=corr, output_names="corrcoef")


    out_f = stroll.apply_func(f_corr)
    assert out_f.columns[0] == f'EDA|TMP__corrcoef__w={int(window)}_s={int(stride)}'
    assert isinstance(out_f.index, pd.DatetimeIndex)

def test_stroll_mixed_input_dtypes(dummy_data):
    # Note: we donnot reset the index of the EDA data to int-index
    eda_data = dummy_data['EDA'].reset_index(drop=True)
    tmp_data = dummy_data['TMP'].reset_index(drop=True)

    with pytest.raises(ValueError):
        stroll = StridedRollingFactory.get_segmenter(
            data=[eda_data, tmp_data],
            window=pd.Timedelta(seconds=300),
            stride=pd.Timedelta(seconds=100)
        )