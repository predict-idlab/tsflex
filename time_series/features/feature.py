"""Contains Feature and MultipleFeature class."""

import itertools
from typing import Callable, List, Union

from .function_wrapper import NumpyFuncWrapper


class FeatureDescriptor:
    """A FeatureDescriptor object, containing all feature information."""

    def __init__(
        self,
        function: Union[NumpyFuncWrapper, Callable],
        key: str,
        window: int,
        stride: int,
    ):
        """Create a FeatureDescriptor object.

        Parameters
        ----------
        function : Union[NumpyFuncWrapper, Callable]
            The `function` that calculates this feature.
        key : str
            The key (name) of the signal where this feature needs to be calculated on.
            This allows to process multivariate series.
        window : int
            The window size on which this feature will be applied, expressed in the
            number of samples from the input signal.
        stride : int
            The stride of the window rolling process, also as a number of samples of the
            input signal.

        Raises
        ------
        TypeError
            Raised when the `function` is not an instance of Callable or
            NumpyFuncWrapper.

        """
        self.key = key
        self.window = window
        self.stride = stride

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
        signal_keys: Union[str, List[str]],
        functions: List[Union[NumpyFuncWrapper, Callable]],
        windows: Union[int, List[int]],
        strides: Union[int, List[int]],
    ):
        """Create a MultipleFeatureDescriptors object.

        Create a list of features from **all** combinations of the given parameter
        lists. Total number of created Features will be:
        len(keys)*len(functions)*len(windows)*len(strides).

        Parameters
        ----------
        signal_keys : Union[str, List[str]],
            All the signal keys.
        functions : List[Union[NumpyFuncWrapper, Callable]]
            The functions, can be either of both types (even in a single array).
        windows : Union[int, List[int]],
            All the window sizes.
        strides : Union[int, List[int]],
            All the strides.

        """
        # convert all types to list
        to_list = lambda x: [x] if not isinstance(x, list) else x
        signal_keys = to_list(signal_keys)
        windows = to_list(windows)
        strides = to_list(strides)

        self.feature_descriptions = []
        # iterate over all combinations
        combinations = [functions, signal_keys, windows, strides]
        for function, key, window, stride in itertools.product(*combinations):
            self.feature_descriptions.append(
                FeatureDescriptor(function, key, window, stride)
            )
