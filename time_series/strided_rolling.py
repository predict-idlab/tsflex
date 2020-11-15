# -*- coding: utf-8 -*-
"""
    *******************
    strided_rolling.py
    *******************

    This module contains a (rather) fast implementation of a strided rolling window
"""

__author__ = 'Vic Degraeve, Jonas Van Der Donckt, Jeroen Van Der Donckt'

from pathos.multiprocessing import ProcessPool
from typing import List, Callable, Union, Dict

import numpy as np
import pandas as pd

from .features import NumpyFeatureCalculation
from .function import NumpyFuncWrapper

import dill as pickle
pickle.settings['recurse'] = True


class StridedRolling:
    """Custom sliding window with stride for pandas DataFrames"""

    def __init__(self, df: pd.DataFrame, window: int, stride: int):
        """
        :param df: DataFrame to slide over, the index must be a (time-zone-aware) date_time object
        :param window: Sliding window length in samples
        :param stride: Step/stride length in samples
        """
        # construct the (expanded) sliding window-stride array
        self.time_indexes = df.index[:-window + 1][::stride]
        self.strided_vals = {}
        for col in df.columns:
            self.strided_vals[col] = sliding_window(df[col], window=window, stride=stride)

    def apply(self, np_func: Callable, return_df: bool = True) -> Union[Dict[str, list], pd.DataFrame]:
        """Applies a function to every strided window and returns the merged outputs in either a new DataFrame
        or a dict

        .. note::
            * As np_funx
            * This only works for a one-to-one mapping!

        :param np_func: Function taking a numpy array as first argument and returning a new numpy array,
            np_func must thus not be a
        :param return_df: If true, a DataFrame will be returned, otherwise a dict will be returned
        :return: Either The merged output of the function applied to every column in a new DataFrame or a dict
        """
        feat_out = {
            col + '_' + np_func.__name__: np.apply_along_axis(np_func, axis=-1, arr=self.strided_vals[col])
            for col in self.strided_vals.keys()
        }
        return pd.DataFrame(index=self.time_indexes, data=feat_out) if return_df else feat_out

    def apply_funcs(self, funcs: List[Union[NumpyFeatureCalculation, NumpyFuncWrapper]], parallel=True) -> pd.DataFrame:
        """Applies a Feature-calculation function to every window

        .. note::
            Every item in the funcs list will thus need to have the same window/stride properties as this instance

        :param funcs: The list of functions which will be applied
        :return: The merged DataFrame
        """
        # TODO: maybe this can be sped up -> also look into memory expansion
        if parallel:
            with ProcessPool() as pool:
                out = pool.map(self.apply_func, funcs, [False]*len(funcs))
            feat_out = {k: v for func_dict in out for k, v in func_dict.items()}
        else:
            feat_out = {k: v for func in funcs for k, v in self.apply_func(func, return_df=False).items()}
        return pd.DataFrame(index=self.time_indexes, data=feat_out)

    def apply_func(self, np_func: Union[NumpyFuncWrapper, NumpyFeatureCalculation], return_df=True) \
            -> Union[Dict[str, list], pd.DataFrame]:
        """Applies a Numpy-function to the

        .. note::
            this works for one-to-many mapping (as both NumpyFuncWrapper and NumpyFeatureCalculation have the
            col_names property)

        :param np_func: The Callable (wrapped) function which will be applied
        :param return_df: If true, a DataFrame will be returned, otherwise a dict will be returned
        :return: Either The merged output of the function applied to every column in a new DataFrame or a dict
        """
        feat_out = {}
        feat_names = np_func.get_col_names()
        for col in self.strided_vals.keys():
            out = np.apply_along_axis(np_func, axis=-1, arr=self.strided_vals[col])
            if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
                assert len(feat_names) == 1
                feat_out[col + '_' + feat_names[0]] = out.flatten()
            if out.ndim == 2 and out.shape[1] > 1:
                assert len(feat_names) == out.shape[1]
                for col_idx in range(out.shape[1]):
                    feat_out[col + '_' + feat_names[col_idx]] = out[:, col_idx]
        return pd.DataFrame(index=self.time_indexes, data=feat_out) if return_df else feat_out


def sliding_window(series: pd.Series, window: int, stride=1, axis=-1) -> np.ndarray:
    """Calculate a strided sliding window over a series.

    :param series: Pandas series to slide over
    :param window: Sliding window length in samples
    :param stride: Step/stride length in samples. Defaults to 1.
    :param axis: The axis to slide over. Defaults to the last axis.
    """
    # TODO: het werkt op DataFrame als je axis = 0 -> wrapper code errond
    data = series.values
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")
    if stride < 1:
        raise ValueError("Step size may not be zero or negative")
    if window > data.shape[axis]:
        raise ValueError("Sliding window size may not exceed size of selected axis")

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stride - window / stride + 1).astype(int)
    shape.append(window)

    strides = list(data.strides)
    strides[axis] *= stride
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return strided
