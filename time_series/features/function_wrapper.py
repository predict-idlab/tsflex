"""Object-oriented representation of a function."""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"

from typing import List, Union, Callable
import types
import warnings

import numpy as np


class NumpyFuncWrapper:
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
        if isinstance(func, types.LambdaType) and func.__name__ == "<lambda>":
            warnings.warn(
                f"\nFunction: {output_names} is a lambda, thus not pickle-able.\n"
                "This might give problems with multiprocessing."
            )

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

    def __repr__(self) -> str:
        """Return repr string."""
        return (
            f"{self.__class__.__name__}({self.func.__name__}, {self.output_names},"
            f" {self.kwargs})"
        )

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Call wrapped function with passed data."""
        return self.func(data, **self.kwargs)
