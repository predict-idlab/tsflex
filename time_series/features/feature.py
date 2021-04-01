"""Contains Feature and MultipleFeature class."""

import itertools
from typing import Callable, List, Union

import pandas as pd

from .function_wrapper import NumpyFuncWrapper


class Feature:
    """A Feature object, containing all feature information."""

    def __init__(
            self,
            function: Union[NumpyFuncWrapper, Callable],
            key: str,
            window: int,
            stride: int,
    ):
        """Create a Feature object.

        Parameters
        ----------
        function : Union[NumpyFuncWrapper, Callable]
            The `function` that calculates this feature.
        key : str
            The key (name) of the signal where this feature needs to be calculated on.
        window : int
            The window size on which this feature will be applied, expressed in the
            number of samples from the input signal.
        stride : int
            The stride of the window rolling process, also as a number of samples of the
            input signal.


        Raises
        ------
        TypeError
            Raise a TypeError when the `function` is not an instance of
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

        # The output of the feature (actual feature data)
        self._output = None

    @property
    def output(self) -> pd.DataFrame:
        """Get the output data for this feature.

        Todo
        ----
        Look into trade-off of storing output data in a Dict[str, pd.DataFrame].
        This will most likely imply that the `apply_func` method of the StridedRolling
        class needs also to be changed.

        Returns
        -------
        pd.DataFrame
            The output data of this Feature, stored in a DataFrame.
            The DataFrame's column-names have the format:
                `<signal_col_name>_<feature_name>__w=<window>_s=<stride>`.

        """
        return self._output

    @output.setter
    def output(self, output: pd.DataFrame):
        # TODO check if the DataFrame columns match the expected FuncWrapper outputs.
        self._output = output

    def __repr__(self) -> str:
        """Representation string of Feature."""
        return (
            f"{self.__class__.__name__}({self.key}, {self.window}, {self.stride},"
            f" {self.function}, {self.output})"
        )


class MultipleFeatures:
    """Create multiple Feature objects."""

    def __init__(
            self,
            signal_keys: List[str],
            functions: List[Union[NumpyFuncWrapper, Callable]],
            windows: List[int],
            strides: List[int],
    ):
        """Create a MultipleFeatures object.

        Create a list of features from **all** combinations of the given parameter
        lists. Total number of created Features will be:
        len(keys)*len(functions)*len(windows)*len(strides).

        Parameters
        ----------
        signal_keys : List[str]
            All the signal keys.
        functions : List[Union[NumpyFuncWrapper, Callable]]
            The functions, can be either of both types (even in a single array).
        windows : List[int]
            All the window sizes.
        strides : List[int]
            All the strides.

        """
        self.features = []
        # iterate over all combinations
        combinations = [functions, signal_keys, windows, strides]
        for function, key, window, stride in itertools.product(*combinations):
            self.features.append(Feature(function, key, window, stride))
