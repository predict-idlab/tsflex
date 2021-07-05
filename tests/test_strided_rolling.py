"""Tests for the strided rolling class"""

from .utils import dummy_data
from tsflex.features.strided_rolling import StridedRolling
from tsflex.features import NumpyFuncWrapper
from tsflex.utils.time import parse_time_arg
import numpy as np


def test_bound_types(dummy_data):
    def corr(s1, s2):
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        return np.corrcoef(s1, s2)[0][-1].astype(s1.dtype)
    f_corr = NumpyFuncWrapper(func=corr, output_names="corrcoef")
    window = parse_time_arg('30s')
    stride = parse_time_arg('30s')
    for bound_method in ['inner', 'outer', 'first']:
        stroll = StridedRolling(
            data=dummy_data[['EDA', 'TMP']],
            window=window,
            stride=stride,
            bound_method=bound_method
        )

        out_f = stroll.apply_func(f_corr)
        assert out_f.columns[0] == 'EDA|TMP__corrcoef__w=30s_s=30s'
