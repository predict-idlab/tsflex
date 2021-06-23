"""Tests for the tsflex.chunking submodule."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

from datetime import datetime

import numpy as np
import pandas as pd

from tsflex.chunking import chunk_data


def test_chunking_univariate_continuous():
    # create some dummy data
    series = pd.Series(
        index=pd.date_range(datetime.now(), periods=10_000, freq='1s'),
        data=np.ones(10_000)
    ).rename('1hz_series')
    out = chunk_data(data=series, fs_dict={"1hz_series": 1}, copy=True)
    assert len(out) == 1
    assert len(out[0][0]) == 10_000
    # as we've returned a copy -> we should not change the original series
    out[0][0][:10] = 0
    assert all(series == 1)


def test_chunking_univariate_continuous_view():
    # create some dummy data
    series = pd.Series(
        index=pd.date_range(datetime.now(), periods=10_000, freq='1s'),
        data=np.ones(10_000)
    ).rename('1hz_series')
    out = chunk_data(data=series, fs_dict={"1hz_series": 1}, copy=False)
    assert len(out) == 1
    assert len(out[0][0]) == 10_000

    # as we've returned a view and not a copy -> must we also change the original series
    out[0][0][:10] = 0
    assert all(series[:10] == 0)
    assert all(series[10:] == 1)


def test_chunking_univariate_too_small_view():
    # create some dummy data
    series = pd.Series(
        index=pd.date_range(datetime.now(), periods=10_000, freq='1s'),
        data=np.ones(10_000)
    ).rename('1hz_series')
    out = chunk_data(data=series[:1], fs_dict={"1hz_series": 1}, copy=False,
                     verbose=True)
    assert len(out) == 0


def test_chunking_multivariate_continuous():
    # create some dummy data
    hz_series = pd.Series(
        index=pd.date_range(datetime.now(), periods=10_000, freq='1s'),
        data=np.ones(10_000)
    ).rename('1hz_series')

    twohz_series = pd.Series(
        index=pd.date_range(datetime.now(), periods=20_000, freq='500ms'),
        data=np.ones(20_000)
    ).rename('2hz_series')
    out = chunk_data(data=[hz_series, twohz_series],
                     fs_dict={"1hz_series": 1, "2hz_series": 2}, copy=True)
    assert len(out) == 1
    assert len(out[0][0]) == 10_000
    assert len(out[0][1]) == 20_000

    # as we've returned a copy -> we should not change the original series
    out[0][0][:10] = 0
    assert all(hz_series == 1)

    # --------- SUB CHUNKS ------------
    # test the sub_chunk_maring and max_chunk duration
    out = chunk_data(
        data=[hz_series, twohz_series],
        fs_dict={"1hz_series": 1, "2hz_series": 2},
        copy=True,
        max_chunk_dur_s=60 * 60,
        sub_chunk_overlap_s=30,
        verbose=True
    )

    # 166 minutes of data -> 3*60 = 180 and is smaller than 166
    assert len(out) == 3
    # assert that, for the middle part -> the sub-chunk time-range ~= 30
    assert (
            out[1][1].index[-1] - out[1][1].index[0] - pd.Timedelta('1hour 1min') < \
            pd.Timedelta('1second')
    )
    # last start minus earliest stop must be ~= 2 times sub_chunk_overlap_s i.e. 1 min
    assert out[0][1].index[-1] - out[1][0].index[0] - pd.Timedelta(
        '1min') < pd.Timedelta('1sec')

    # --------- SUB CHUNKS ------------
    # test the sub_chunk_marging and max_chunk duration & min_chunk_duration
    # everything should remain the same
    out = chunk_data(
        data=[hz_series, twohz_series],
        fs_dict={"1hz_series": 1, "2hz_series": 2},
        copy=True,
        max_chunk_dur_s=60 * 60,
        sub_chunk_overlap_s=30,
        min_chunk_dur_s=30,
        verbose=True
    )

    # 166 minutes of data -> 3*60 = 180 and is smaller than 166
    assert len(out) == 3
    # assert that, for the middle part -> the sub-chunk time-range ~= 30
    assert (
            out[1][1].index[-1] - out[1][1].index[0] - pd.Timedelta('1hour 1min') < \
            pd.Timedelta('1second')
    )
    # last start minus earliest stop must be ~= 2 times sub_chunk_overlap_s i.e. 1 min
    assert out[0][1].index[-1] - out[1][0].index[0] - pd.Timedelta(
        '1min') < pd.Timedelta('1sec')


def test_chunking_univariate_gaps():
    # create some dummy data
    series = pd.Series(
        index=pd.date_range(datetime.now(), periods=10_000, freq='1s'),
        data=np.ones(10_000)
    ).rename('1hz_series')

    series = series.drop(series.index[10:20])
    series = series.drop(series.index[1000:1003])

    # no chunk limitations -> so we should get 3 chunks and in total, the sum of all
    # outs must be len(series)
    out = chunk_data(data=series, fs_dict={"1hz_series": 1}, copy=True, verbose=True)
    assert len(out) == 3
    assert sum(len(out[i][0]) for i in range(len(out))) == len(series)

    # min chunk duration -> will now disregard the first chunk
    out = chunk_data(data=series, fs_dict={"1hz_series": 1}, copy=True, verbose=True,
                     min_chunk_dur_s=30)
    assert len(out) == 2

    # drop an additional chunk, making the first and chunk which orignates from 0:10 and
    # will now be split up in 0:2 and 8:10 both too small -> both chunks will be dropped
    # len = 2
    series = series.drop(series.index[2:8])
    out = chunk_data(data=series, fs_dict={"1hz_series": 1}, copy=True, verbose=True)
    assert len(out) == 2
