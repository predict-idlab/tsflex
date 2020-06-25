# -*- coding: utf-8 -*-
"""
    ************
    function.py
    ************
    
    Created at 14/06/20

    Object-oriented representation of a function
"""

__author__ = 'Jonas Van Der Donckt, Jeroen Van Der Donckt'

from abc import ABC, abstractmethod
from typing import List, Union, Callable

import numpy as np


class FuncWrapper(ABC):
    """Abstract class which extends a Callable function with additional logic (output col names, kwargs)"""

    def __init__(self, func: Callable, col_names: Union[List[str], str] = None, **kwargs):
        """Wraps a function and save the additional metadata

        :param func: The function which is wrapped
        :param col_names: The output column names of the function
        :param kwargs: The additional keyword arguments for the function
        """
        self.func = func
        self.kwargs: dict = kwargs
        col_names = func.__name__ if col_names is None else col_names
        self.col_names = [col_names] if isinstance(col_names, str) else col_names

    def get_col_names(self) -> List[str]:
        return list(self.col_names)

    def __str__(self) -> str:
        return f'{self.func.__name__} - kwargs: {self.kwargs} - col_names: {self.col_names}'

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def __call__(self, data) -> any:
        """Executes the function, sub-classes must have implemented this method"""
        raise NotImplementedError


class NumpyFuncWrapper(FuncWrapper):
    """Function wrapper which takes a numpy array as input"""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.func(data) if self.kwargs is None else self.func(data, **self.kwargs)
