"""NumpyFuncWrapper class for object-oriented representation of a function."""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

from typing import Callable, List, Union, Any

import numpy as np

from .. import __pdoc__
from ..utils.classes import FrozenClass

__pdoc__['NumpyFuncWrapper.__call__'] = True


class NumpyFuncWrapper(FrozenClass):  # TODO: waarom niet gewoon FuncWrapper?
    """Numpy function wrapper.

    A Numpy function wrapper which takes a numpy array as input and returns a numpy
    array. It also defines the names of the function outputs.

    Parameters
    ----------
    func : Callable
        The wrapped function.
    output_names : Union[List[str], str], optional
        The name of the outputs of the function, by default None.

    Raises
    ------
    TypeError
        Raised when the `output_names` cannot be set.

    """

    def __init__(
        self, func: Callable, output_names: Union[List[str], str] = None, **kwargs
    ):
        """Create NumpyFuncWrapper instance."""
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

        self._freeze()

    def __repr__(self) -> str:
        """Return repr string."""
        return (
            f"{self.__class__.__name__}({self.func.__name__}, {self.output_names},"
            f" {self.kwargs})"
        )

    def __call__(self, *series: np.ndarray) -> Any:
        """Call wrapped function with passed data.

        Parameters
        ---------
        *series : np.ndarray
            The (multiple) input series for the function.

        Returns
        -------
        Any
            The function its output for the passed series.

        """
        return self.func(*series, **self.kwargs)
