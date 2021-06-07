"""FeatureDescriptor and MultipleFeatureDescriptors class for creating time-series features."""

import itertools
from typing import Callable, List, Union, Tuple

import pandas as pd

from .function_wrapper import NumpyFuncWrapper


class FeatureDescriptor:
    """A FeatureDescriptor object, containing all feature information."""

    def __init__(
        self,
        function: Union[NumpyFuncWrapper, Callable],
        key: Union[str, Tuple[str]],
        window: Union[int, str, pd.Timedelta],
        stride: Union[int, str, pd.Timedelta],
    ):
        """Create a FeatureDescriptor object.

        Notes
        -----
        * For each function - input(-series) - window - stride combination, one needs
          to create a distinct `FeatureDescriptor`. Hence it is more convenient to create
          a `MultipleFeatureDescriptors` when `function` - `window` - `stride`
          combination should be applied on various input-series (combinations).
        * When `function` takes multiple series (i.e., arguments) as input, these are
          merged (based on the index) before applying the function. Thus make sure to 
          use time-based window and stride arguments in this constructor to avoid
          unexpected behavior. If the indexes of the series are not exactly the same,
          there will be `NaN`s after merging into a dataframe, hence make sure that the
          `function` can deal with this!

        Parameters
        ----------
        function : Union[NumpyFuncWrapper, Callable]
            The function that calculates this feature.
        key : Union[str,Tuple[str]]
            The name(s) of the series on which this feature (its `function`) needs to
            be calculated. \n
            * If `function` has just one series as argument, `key` should be a string
              containing the name of that series. We call such a function a
              *single input-series function*.
            * If `function` has multiple series, this argument should be a tuple of
              strings containing the ordered names of those series. When calculating
              this feature, the exact order of series is used as provided by the tuple.
              We call such a function a *multi input-series function*.
            TODO: assumption van merge nr pd.DataFrame in stroll? => signals zelfde freq / gaps ?
        window :  Union[int, str, pd.Timedelta]
            The window size, this argument supports multiple types.
            If the type is an int, it represents the number of samples of the input
            series. If the window's type is a `pd.Timedelta`, the window size represents
            the window-time. If a `str`, it represents a window-time-string.
        stride :  Union[int, str, pd.Timedelta]
            The stride of the window rolling process, supports multiple types. \n
            * If the type is int, it represents the number of samples of the input
              series that will be rolled over.
            * If the type is `pd.Timedelta`, it represents the stride-roll timedelta.
            * If a type is str, it represents a stride-roll-time-string.

        Note
        ----
        Later on, the (not-int) time-based window-stride parameters, are converted into
        ints in the `StridedRolling` class. This time -> int conversion implies three
        things:

        1. The time -> int conversion will be done at inference time. Hence, the
            converted int will be dependent of the inference-time `series-argument`'s
            frequency (for which this feature will be extracted).
        2. This inference time conversion also implies that **each** series on
            which the features will be extracted **must contain** a frequency.
            **So no gaps are allowed in these series!**
        3. The time **will be converted to an int**, and as this is achieved by dividing
            the `pd.TimeDelta` through the series' inferred freq timedelta.

        Raises
        ------
        TypeError
            Raised when the `function` is not an instance of Callable or
            NumpyFuncWrapper.

        See Also
        --------
        The `StridedRolling` class, as the window-stride (time) conversion takes
        place there.

        """
        to_tuple = lambda x: tuple([x]) if isinstance(x, str) else x
        self.key = to_tuple(key)  # TODO: wrm per se allemaal tuple?
        self.window = FeatureDescriptor._parse_time_arg(window)
        self.stride = FeatureDescriptor._parse_time_arg(stride)

        # Order of if statements is important!
        if isinstance(function, NumpyFuncWrapper):
            self.function = function
        elif isinstance(function, Callable):
            self.function = NumpyFuncWrapper(function)
        else:
            raise TypeError(
                "Expected feature function to be a `NumpyFuncWrapper` but is a"
                f" {type(function)}."
            )

    def is_single_series_func(self) -> bool:  # TODO: dit nodig?
        """Return whether this feature is a single series function.

        A single series function is a `function` that takes single series as input.

        Returns
        -------
        bool
            Whether the feature its `function` takes a single series as input.
        """
        return len(self.key) == 1

    @staticmethod
    def _parse_time_arg(arg: Union[int, str, pd.Timedelta]) -> Union[int, pd.Timedelta]:
        """Parse the `window`/`stride` arg into a fixed set of types.

        Parameters
        ----------
        arg : Union[int, str, pd.Timedelta]
            The arg that will be parsed. If the type is either an `int` or
            `pd.Timedelta`, nothing will happen. If the type is a `str`, `arg` should
            represent a time-string, and will be converted to a `pd.Timedelta`.

        Returns
        -------
        Union[int, pd.Timedelta]
            Either an int or `pd.Timedelta`, dependent on the arg-input.

        Raises
        ------
        TypeError
            Raised when `arg` is not an instance of `int`, `str`, `pd.Timedelta`
        """
        if isinstance(arg, int) or isinstance(arg, pd.Timedelta):
            return arg
        elif isinstance(arg, str):
            return pd.Timedelta(arg)
        raise TypeError(f"arg type {type(arg)} is not supported!")

    def __repr__(self) -> str:
        """Representation string of Feature."""
        return f"{self.__class__.__name__}({self.key}, {self.window}, {self.stride})"

    def _func_str(self) -> str:
        if isinstance(self.function, NumpyFuncWrapper):
            f_name = self.function
        else:
            f_name = self.function.__name__
        return f"{self.__class__.__name__} - func: {str(f_name)}"


class MultipleFeatureDescriptors:
    """Create multiple FeatureDescriptor objects."""

    def __init__(
        self,
        functions: List[Union[NumpyFuncWrapper, Callable]],
        keys: Union[str, Tuple[str], List[str], List[Tuple[str]]],
        windows: Union[int, str, pd.Timedelta, List[Union[int, str, pd.Timedelta]]],
        strides: Union[int, str, pd.Timedelta, List[Union[int, str, pd.Timedelta]]],
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
            Note: when passing a list of `key`s, all `key`s in this
            list should have the same type, i.e, either \n
            * all a str
            * or, all a tuple with same length. \n
            Read more about the `key` argument in `FeatureDescriptor`.
        windows : Union[int, str, pd.Timedelta, List[Union[int, str, pd.Timedelta]]],
            All the window sizes.
        strides : Union[int, str, pd.Timedelta, List[Union[int, str, pd.Timedelta]]],
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

        self.feature_descriptions = []
        # iterate over all combinations
        combinations = [functions, keys, windows, strides]
        for function, key, window, stride in itertools.product(*combinations):
            self.feature_descriptions.append(
                FeatureDescriptor(function, key, window, stride)
            )
