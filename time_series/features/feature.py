"""Contains Feature and MultipleFeature class."""

import itertools
from typing import Callable, List, Union, Tuple

import pandas as pd

from .function_wrapper import NumpyFuncWrapper


class FeatureDescriptor:
    """A FeatureDescriptor object, containing all feature information."""

    def __init__(
        self,
        function: Union[NumpyFuncWrapper, Callable],
        func_input: Union[str, Tuple[str]],  # TODO: is het wel goed om `key` te exposen aan de end-user? => veranderd nr func_input
        window: Union[int, str, pd.Timedelta],
        stride: Union[int, str, pd.Timedelta],
    ):
        """Create a FeatureDescriptor object.

        Notes
        -----
        * For each function - input(-signals) - window - stride combination, one needs 
          to create a distinct FeatureDescriptor. Hence it is more convenient to create
          a `MultipleFeatureDescriptors` when `function` - `window` - `stride`
          combination should be applied on various input-signals.
        * When `function` takes multiple signals (i.e., arguments) as input, these are
          merged (based on the index) before applying the function. Thus make sure to 
          use time-based window and stride arguments in this constructor to avoid
          unexpected behavior. If the indexes of the signals are not exactly the same, 
          there will be `NaN`s after merging into a dataframe, hence make sure that the
          `function` can deal with this!

        Parameters
        ----------
        function : Union[NumpyFuncWrapper, Callable]
            The `function` that calculates this feature.
        func_input : Union[str,Tuple[str]]
            The name(s) of the signal(s) on which this feature (its `function`) needs to
            be calculated.
            If `function` uses just one signal, this argument should be a string
            containing the name of that signal.
            If `function` uses multiple signals, this argument should be a tuple of
            strings containing the ordered names of those signals. When calculating this
            feature, the exact order of signals is used as  provided by that tuple.
        window :  Union[int, str, pd.Timedelta]
            The window size, this argument supports multiple types.
            If the type is an int, it represents the number of samples of the input
            signal. If the window's type is a `pd.Timedelta`, the window size represents
            the window-time. If a `str`, it represents a window-time-string.
        stride :  Union[int, str, pd.Timedelta]
            The stride of the window rolling process, supports multiple types.
            If the type is an int, it represents the number of samples of the input
            signal that will be rolled over. If the stride's type is a `pd.Timedelta`,
            it represents the stride-roll timedelta. If a `str`, it represents a
            stride-roll-time-string.

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
            the `pd.TimeDelta` through the signal's inferred freq timedelta.

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
        self.key = to_tuple(func_input) # TODO: wrm per se allemaal tuple?
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

    def is_single_series_func(self) -> bool: # TODO: dit nodig?
        """Return whether this feature takes a single series as input.

        Returns
        -------
        bool
            Whether the feature its function takes a single series as input.
        """
        return len(self.key) == 1

    @staticmethod
    def _parse_time_arg(arg: Union[int, str, pd.Timedelta]) -> Union[int, pd.Timedelta]:
        """Parse the `window`/`stride` arg into a fixed set of types.

        Parameters
        ----------
        arg: Union[int, str, pd.Timedelta]
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
        # TODO: perhaps signal_keys was better 
        func_inputs: Union[str, Tuple[str], List[str], List[Tuple[str]]], # TODO: is het wel goed om `key` te exposen aan de end-user? => veranderd nr func_input
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
        func_inputs : Union[str, Tuple[str], List[str], List[Tuple[str]]],
            All the function inputs. A single function input is/are the name(s) of the
            signal(s) on which every function in `functions` needs to be calculated.
            # TODO: assumption van merge nr pd.DataFrame in stroll? => signals zelfde freq / gaps ?
        windows : Union[int, str, pd.Timedelta, List[Union[int, str, pd.Timedelta]]],
            All the window sizes.
        strides : Union[int, str, pd.Timedelta, List[Union[int, str, pd.Timedelta]]],
            All the strides.

        """
        # Convert all types to list
        to_list = lambda x: [x] if not isinstance(x, list) else x
        signal_keys = to_list(func_inputs)
        windows = to_list(windows)
        strides = to_list(strides)

        # Assert that function inputs are from the same length
        to_tuple = lambda x: tuple([x]) if isinstance(x, str) else x
        assert all(
            [len(to_tuple(signal_keys[0])) == len(to_tuple(key)) for key in signal_keys]
        )

        self.feature_descriptions = []
        # iterate over all combinations
        combinations = [functions, signal_keys, windows, strides]
        for function, key, window, stride in itertools.product(*combinations):
            self.feature_descriptions.append(
                FeatureDescriptor(function, key, window, stride)
            )
