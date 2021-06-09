# -*- coding: utf-8 -*-
"""

FeatureDescriptor and MultipleFeatureDescriptors class for creating time-series
features.

"""

import itertools
from typing import Callable, List, Union, Tuple

import pandas as pd

from .function_wrapper import NumpyFuncWrapper
from ..utils.classes import FrozenClass


class FeatureDescriptor(FrozenClass):
    """A FeatureDescriptor object, containing all feature information."""

    def __init__(
        self,
        function: Union[NumpyFuncWrapper, Callable],
        key: Union[str, Tuple[str]],
        window: Union[float, str, pd.Timedelta],
        stride: Union[float, str, pd.Timedelta],
    ):
        """Create a FeatureDescriptor object.

        Parameters
        ----------
        function : Union[NumpyFuncWrapper, Callable]
            The function that calculates this feature.
        key : Union[str,Tuple[str]]
            The name(s) of the series on which this feature (its `function`) needs to
            be calculated. \n
            * If `function` has just one series as argument, `key` should be a `str`
              containing the name of that series.
            * If `function` has multiple series, this argument should be a `Tuple[str]`,
              containing the ordered names of those series. When calculating
              this feature, the **exact order of series is used as provided by the
              tuple**. We call such a function a *multi input-series function*.
        window : Union[float, str, pd.Timedelta]
            The window size, this argument supports multiple types: \n
            * If the type is an `float`, it represents the series its window-size in
              **seconds**.
            * If the window's type is a `pd.Timedelta`, the window size represents
              the window-time.
            * If a `str`, it represents a window-time-string.
        stride : Union[int, str, pd.Timedelta]
            The stride of the window rolling process, supports multiple types. \n
            * If the type is `float`, it represents the window size in **seconds**
            * If the type is `pd.Timedelta`, it represents the stride-roll timedelta.
            * If a type is `str`, it represents a stride-roll-time-string.

        Notes
        -----
        * For each function - input(-series) - window - stride combination, one needs
          to create a distinct `FeatureDescriptor`. Hence it is more convenient to
          create a `MultipleFeatureDescriptors` when `function` - `window` - `stride`
          _combinations_ should be applied on various input-series (combinations).
        * When `function` takes multiple series (i.e., arguments) as input, these are
          joined (based on the index) before applying the function. If the indexes of
          these series are not exactly the same, it might occur that not all series have
          exactly the same length! Hence,  make sure that the `function` can deal with
           this!

        TODO
        ----
        <Add documentation of how the index/slicing takes place> / which assumptions
        we make.

        Raises
        ------
        TypeError
            Raised when the `function` is not an instance of Callable or
            NumpyFuncWrapper.

        See Also
        --------
        StridedRolling: As the window-stride (time) conversion takes place there.

        https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html#parsing,
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html#pandas-timedelta

        """
        to_tuple = lambda x: tuple([x]) if isinstance(x, str) else x
        self.key: tuple = to_tuple(key)
        self.window = FeatureDescriptor._parse_time_arg(window)
        self.stride = FeatureDescriptor._parse_time_arg(stride)

        # Order of if statements is important (as NumpyFuncWrapper also is a Callable)!
        if isinstance(function, NumpyFuncWrapper):
            self.function = function
        elif isinstance(function, Callable):
            self.function = NumpyFuncWrapper(function)
        else:
            raise TypeError(
                "Expected feature function to be a `NumpyFuncWrapper` but is a"
                f" {type(function)}."
            )

        # construct a function-string
        if isinstance(self.function, NumpyFuncWrapper):
            f_name = self.function
        else:
            f_name = self.function.__name__
        self._func_str: str = f"{self.__class__.__name__} - func: {str(f_name)}"

        self._freeze()

    @staticmethod
    def _parse_time_arg(arg: Union[float, str, pd.Timedelta]) -> pd.Timedelta:
        """Parse the `window`/`stride` arg into a fixed set of types.

        Parameters
        ----------
        arg : Union[float, str, pd.Timedelta]
            The arg that will be parsed. \n
            * If the type is an `int` or `float`, it should represent the timedelta in
              **seconds**.
            * If the type is a `pd.Timedelta`, nothing will happen.
            * If the type is a `str`, `arg` should represent a time-string, and will be
              converted to a `pd.Timedelta`.

        Returns
        -------
        pd.Timedelta
            The parsed time arg

        Raises
        ------
        TypeError
            Raised when `arg` is not an instance of `float`, `int`, `str`, or
            `pd.Timedelta`.

        """
        if isinstance(arg, int) or isinstance(arg, float):
            return pd.Timedelta(seconds=arg)
        elif isinstance(arg, str):
            return pd.Timedelta(arg)
        raise TypeError(f"arg type {type(arg)} is not supported!")

    def __repr__(self) -> str:
        """Representation string of Feature."""
        return f"{self.__class__.__name__}({self.key}, {self.window}, {self.stride})"


class MultipleFeatureDescriptors:
    """Create multiple FeatureDescriptor objects."""

    def __init__(
        self,
        functions: List[Union[NumpyFuncWrapper, Callable]],
        keys: Union[str, Tuple[str], List[str], List[Tuple[str]]],
        windows: Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]],
        strides: Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]],
    ):
        """Create a MultipleFeatureDescriptors object.

        Create a list of features from **all** combinations of the given parameter
        lists. Total number of created Features will be:

            len(func_inputs)*len(functions)*len(windows)*len(strides).

        Parameters
        ----------
        functions : List[Union[NumpyFuncWrapper, Callable]]
            The functions, can be either of both types (even in a single array).
        keys : Union[str, Tuple[str], List[str], List[Tuple[str]]],
            All the function inputs (either a `key` or a list of `key`s).
            A `key` contains the name(s) of the series on which every function in
            `functions` needs to be calculated. Hence, when the function(s) should be
            called on multiple (combinations of) series, one should pass a list of
            `key`s. \n
            **Note**: when passing a list of `key`s, all `key`s in this
            list should have the same type, i.e, either \n
            * all a str
            * or, all a tuple _with same length_. \n
            Read more about the `key` argument in `FeatureDescriptor`.
        windows : Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]],
            All the window sizes.
        strides : Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]],
            All the strides.

        """
        # Convert all types to list
        to_list = lambda x: [x] if not isinstance(x, list) else x
        keys = to_list(keys)
        windows = to_list(windows)
        strides = to_list(strides)

        # Assert that function inputs are from the same length
        to_tuple = lambda x: tuple([x]) if isinstance(x, str) else x
        assert all(
            [len(to_tuple(keys[0])) == len(to_tuple(key)) for key in keys]
        )

        self.feature_descriptions: List[FeatureDescriptor] = []
        # iterate over all combinations
        combinations = [functions, keys, windows, strides]
        for function, key, window, stride in itertools.product(*combinations):
            self.feature_descriptions.append(
                FeatureDescriptor(function, key, window, stride)
            )
