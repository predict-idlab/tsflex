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
from ..utils.data import to_list, to_tuple


class FeatureDescriptor(FrozenClass):
    """A FeatureDescriptor object, containing all feature information.

    Parameters
    ----------
    function : Union[NumpyFuncWrapper, Callable]
        The function that calculates this feature.
        The prototype of the function should match: \n

            function(*series: np.ndarray)
                -> Union[Any, List[Any]]

    series_name : Union[str, Tuple[str]]
        The names of the series on which the feature function should be applied.
        This argument should match the `function` its input; \n
        * If `series_name` is a string (or tuple of a single string), than 
            `function` should require just one series as input.
        * If `series_name` is a tuple of strings, than `function` should
            require `len(tuple)` series as input **and in exactly the same order**
    window : Union[float, str, pd.Timedelta]
        The window size, this argument supports multiple types: \n
        * If the type is an `float`, it represents the series its window size in
            **seconds**.
        * If the window's type is a `pd.Timedelta`, the window size represents
            the window-time.
        * If a `str`, it represents a window-time-string. \n
            .. Note::
                When no time-unit is present in the string, it represents the stride
                size in **seconds**.

    stride : Union[int, str, pd.Timedelta]
        The stride of the window rolling process, supports multiple types: \n
        * If the type is `float`, it represents the stride size in **seconds**
        * If the type is `pd.Timedelta`, it represents the stride-roll timedelta.
        * If a type is `str`, it represents a stride-roll-time-string. \n
            .. Note::
                When no time-unit is present in the string, it represents the stride
                size in **seconds**.

    Notes
    -----
    * For each `function` - `input`(-series) - `window` - stride combination, one needs
      to create a distinct `FeatureDescriptor`. Hence it is more convenient to
      create a `MultipleFeatureDescriptors` when `function` - `window` - `stride`
      _combinations_ should be applied on various input-series (combinations).
    * When `function` takes multiple series (i.e., arguments) as input, these are
      joined (based on the index) before applying the function. If the indexes of
      these series are not exactly the same, it might occur that not all series have
      exactly the same length! Hence,  make sure that the `function` can deal with
      this!
    * For more information about the str-based time args, look into:
      [pandas time delta](https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html#parsing){:target="_blank"}
    <br><br>
    .. todo::
        * Add documentation of how the index/slicing takes place / which
          assumptions we make.
        * Raise error function tries to change values of view due to flag


    Raises
    ------
    TypeError
        Raised when the `function` is not an instance of Callable or
        NumpyFuncWrapper.

    See Also
    --------
    StridedRolling: As the window-stride (time) conversion takes place there.

    """

    def __init__(
        self,
        function: Union[NumpyFuncWrapper, Callable],
        series_name: Union[str, Tuple[str]],
        window: Union[float, str, pd.Timedelta],
        stride: Union[float, str, pd.Timedelta],
    ):
        self.series_name: Tuple[str] = to_tuple(series_name)
        self.window: pd.Timedelta = FeatureDescriptor._parse_time_arg(window)
        self.stride: pd.Timedelta = FeatureDescriptor._parse_time_arg(stride)

        # Order of if statements is important (as NumpyFuncWrapper also is a Callable)!
        if isinstance(function, NumpyFuncWrapper):
            self.function: NumpyFuncWrapper = function
        elif isinstance(function, Callable):
            self.function: NumpyFuncWrapper = NumpyFuncWrapper(function)
        else:
            raise TypeError(
                "Expected feature function to be a `NumpyFuncWrapper` but is a"
                f" {type(function)}."
            )

        # Construct a function-string
        f_name = str(self.function)
        self._func_str: str = f"{self.__class__.__name__} - func: {f_name}"

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
              Note: if there is no time-unit in the string, it should represent the
              timedelta in **seconds**.

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
        try:
            # Cast a string without time units to float (or cast an int to float)
            arg = float(arg)
        except:
            pass

        if isinstance(arg, float):
            return pd.Timedelta(seconds=arg)
        elif isinstance(arg, str):
            return pd.Timedelta(arg)
        raise TypeError(f"arg type {type(arg)} is not supported!")

    def get_required_series(self) -> List[str]:
        """Return all required series names for this feature descriptor.

        Return the list of series names that are required in order to execute the
        feature function.

        Returns
        -------
        List[str]
            List of all the required series names.

        """
        return list(set(self.series_name))        

    def __repr__(self) -> str:
        """Representation string of Feature."""
        return f"{self.__class__.__name__}({self.series_name}, {self.window}, " \
               f"{self.stride})"


class MultipleFeatureDescriptors:
    """Create a MultipleFeatureDescriptors object.

    Create a list of features from **all** combinations of the given parameter
    lists. Total number of created Features will be:

        len(func_inputs)*len(functions)*len(windows)*len(strides).

    Parameters
    ----------
    functions : List[Union[NumpyFuncWrapper, Callable]]
        The functions, can be either of both types (even in a single array).
    series_names : Union[str, Tuple[str], List[str], List[Tuple[str]]]
        The names of the series on which the feature function should be applied.

        This argument should match the `function` its input; \n
        * If `series_names` is a (list of) string (or tuple of a single string),
          than `function` should require just one series as input.
        * If `series_names` is a (list of) tuple of strings, than `function` should
          require `len(tuple)` series as input.

        A list means multiple series (combinations) to extract feature from; \n
        * If `series_names` is a string or a tuple of strings, than `function` will
          be called only once for the series of this argument.
        * If `series_names` is a list of either strings or tuple of strings, than
          `function` will be called for each entry of this list.

        Note: when passing a list as `series_names`, all items in this list should
        have the same type, i.e, either \n
        * all a str
        * or, all a tuple _with same length_. \n
    windows : Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]],
        All the window sizes.
    strides : Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]],
        All the strides.

    """
    def __init__(
        self,
        functions: List[Union[NumpyFuncWrapper, Callable]],
        series_names: Union[str, Tuple[str], List[str], List[Tuple[str]]],
        windows: Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]],
        strides: Union[float, str, pd.Timedelta, List[Union[float, str, pd.Timedelta]]],
    ):
        # Cast functions to NumpyFuncWrapper, this avoids creating multiple
        # NumpyFuncWrapper objects for the same function in the FeatureDescriptor
        def to_np_func_wrapper(f: Callable): 
            return f if isinstance(f, NumpyFuncWrapper) else NumpyFuncWrapper(f)
        functions = [to_np_func_wrapper(f) for f in functions]
        # Convert the series names to list of tuples
        series_names = [to_tuple(names) for names in to_list(series_names)]
        # Assert that function inputs (series) all have the same length
        assert all(
            len(series_names[0]) == len(series_name_tuple)
            for series_name_tuple in series_names
        )
        # Convert the other types to list
        windows = to_list(windows)
        strides = to_list(strides)

        self.feature_descriptions: List[FeatureDescriptor] = []
        # Iterate over all combinations
        combinations = [functions, series_names, windows, strides]
        for function, series_name, window, stride in itertools.product(*combinations):
            self.feature_descriptions.append(
                FeatureDescriptor(function, series_name, window, stride)
            )
