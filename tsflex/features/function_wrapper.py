"""FuncWrapper class for object-oriented representation of a function."""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

import numpy as np
import pandas as pd

from typing import Callable, List, Union, Any, Optional
from ..utils.data import SUPPORTED_STROLL_TYPES
from ..utils.classes import FrozenClass
from .. import __pdoc__

__pdoc__["FuncWrapper.__call__"] = True


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
    vectorized: bool, optional
        Flag indicating whether `func` should be executed vectorized over all the
        segmented windows, by default False.
        .. Info::
            A vectorized function should take one or multiple series that each have the
            shape (nb. segmented windows, window size).
            For example a vectorized version of `np.max` is
            ``FuncWrapper(np.max, vectorized=True, axis=1)``.
        .. Note::
            * A function can only be applied in vectorized manner when the required 
              series are REGULARLY sampled (and have the same index in case of multiple
              required series).
            * The `input_type` should be `np.array` when `vectorized` is True. It does 
              not make sense to use a `pd.Series`, as the index should be regularly
              sampled (see requirement above).
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
        vectorized: bool = False,
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

        assert input_type in SUPPORTED_STROLL_TYPES, "Invalid input_type!"
        assert not (
            vectorized & (input_type is not np.array)
        ), "The input_type must be np.array if vectorized is True!"
        self.input_type = input_type
        self.vectorized = vectorized

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
