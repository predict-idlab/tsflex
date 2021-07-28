"""FuncWrapper class for object-oriented representation of a function."""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import numpy as np
import pandas as pd

from typing import Callable, List, Union, Any, Optional
from ..utils.data import SUPPORTED_STROLL_TYPES
from ..utils.classes import FrozenClass
from .. import __pdoc__

__pdoc__['FuncWrapper.__call__'] = True


class FuncWrapper(FrozenClass):
    """Function wrapper.

    A function wrapper which takes a numpy array / pandas series as input and returns 
    one or multiple values. It also defines the names of the function outputs, and 
    stores the function its keyword arguments.

    Parameters
    ----------
    func : Callable
        The wrapped function.
    output_names : Union[List[str], str], optional
        The name of the outputs of the function, by default None.
    input_type: Union[np.array, pd.Series], optional
        The input type that the function requires (either np.array or pd.Series), by 
        default np.array.
        .. Note:: 
            Make sure to only set this argument to pd.Series if the function requires
            a pd.Series, since pd.Series strided-rolling is significantly less efficient. 
            For a np.array it is possible to create very efficient views, but there is no 
            such thing as a pd.Series view. Thus, for each stroll, a new series is created.
    **kwargs: dict, optional
        Keyword arguments which will be also passed to the `function`

    Raises
    ------
    TypeError
        Raised when the `output_names` cannot be set.

    """

    def __init__(
        self,
        func: Callable,
        output_names: Optional[Union[List[str], str]] = None,
        input_type: Optional[Union[np.array, pd.Series]] = np.array,
        **kwargs,
    ):
        """Create FuncWrapper instance."""
        self.func = func
        self.kwargs: dict = kwargs

        if isinstance(output_names, list):
            self.output_names = output_names
        elif isinstance(output_names, str):
            self.output_names = [output_names]
        elif not output_names:
            self.output_names = [self.func.__name__]
        else:
            raise TypeError(f"`output_names` is unexpected type {type(output_names)}")

        assert input_type in SUPPORTED_STROLL_TYPES
        self.input_type = input_type

        self._freeze()

    def __repr__(self) -> str:
        """Return repr string."""
        return (
            f"{self.__class__.__name__}({self.func.__name__}, {self.output_names},"
            f" {self.kwargs})"
        )

    def __call__(self, *series: Union[np.ndarray, pd.Series]) -> Any:
        """Call wrapped function with passed data.

        Parameters
        ---------
        *series : Union[np.ndarray, pd.Series]
            The (multiple) input series for the function.

        Returns
        -------
        Any
            The function output for the passed series.

        """
        return self.func(*series, **self.kwargs)
