# -*- coding: utf-8 -*-
"""
    ***********
    feature.py
    ***********

    Withholds code for to represent manners of calculating features
"""
__author__ = 'Jonas Van Der Donckt'

from typing import List, Tuple, Callable

import numpy as np

from ..function import NumpyFuncWrapper


class NumpyFeatureCalculation:
    """Wrapper class around a feature calculation function"""

    def __init__(self, win_size: int, stride: int, func: Callable):
        """
        :param int win_size: The number of sensor observations that each window contain
        :param int stride: Stride size in # of observations
        :param func: The feature calculation func, takes a np array as input and outputs a np array
        """
        self.win_size = win_size
        self.stride = stride
        self.func = func
        self.col_names = func.get_col_names() if isinstance(func, NumpyFuncWrapper) else [func.__name__]

    def get_win_stride(self) -> Tuple[int, int]:
        """Return the (absolute) window_size and stride in (#of samples)"""
        return self.win_size, self.stride

    def get_col_names(self) -> List[str]:
        """Return the column names of the feature"""
        return self.col_names

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        """Cal(l)culates the feature(s)

        :param arr: Array of correct win_size
        :return: The calculated feature(s)
        """
        return self.func(arr)

    def __repr__(self) -> str:
        f_name = self.func if isinstance(self.func, NumpyFuncWrapper) else self.func.__name__
        return f'{self.__class__.__name__} - func: {str(f_name)}'

    def __str__(self) -> str:
        return self.__repr__()
