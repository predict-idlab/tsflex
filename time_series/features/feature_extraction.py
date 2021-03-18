# -*- coding: utf-8 -*-
"""
    ************
    feature_extraction.py
    ************


"""
__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

from typing import List, Union, Callable, Dict

import dill as pickle
import pandas as pd

from .strided_rolling import StridedRolling
from ..function import NumpyFuncWrapper

# TODO: Why is this?
pickle.settings["recurse"] = True

# Food for thought:
#  - Create a subclass of pd.Series which enforces that the index is a DateTime?
#  - Use time based slicing instead of number (of samples) based.


class Feature:
    """A Feature object, containing all feature information."""

    def __init__(self, function: NumpyFuncWrapper, key: str, window: int, stride: int):
        """Create a Feature object.

        Parameters
        ----------
        function : NumpyFuncWrapper
            The `function` that calculates this feature
        key : str
            The key (name) of the signal where this feature needs to be calculated on.
        window : int
            The window size on which this feature will be applied, expressed in the
            number of sample from the input signal.
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
        if isinstance(function, NumpyFuncWrapper):
            self.function = function
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

        Returns
        -------
        Dict[str, pd.Series]
            The output data of this feature, the dict key are the expected outputs for
            the feature and the items the actual Series.
        """
        return self._output

    @output.setter
    def output(self, output: pd.DataFrame):
        # TODO check if the DataFrame columns match the expected FunctWrapper outputs.
        self._output = output

    def __repr__(self) -> str:
        """Representation string of Feature."""
        return (
            f"{self.__class__.__name__}({self.key}, {self.window}, {self.stride},"
            f" {self.function}, {self.output})"
        )


class MultipleFeatures:
    """Expands given feature parameter lists to multiple Feature objects."""

    def __init__(
        self,
        signal_keys: List[str],
        functions: Union[List[NumpyFuncWrapper], List[Callable]],
        windows: List[int],
        strides: List[int],
    ):
        """Create a MultipleFeatures object.

        A list of features will be created with a combination of all the the given
        parameter lists.
        Parameters
        ----------
        signal_keys : List[str]
            Signal keys
        functions : Union[List[NumpyFuncWrapper], List[Callable]]
            The functions
        windows : List[int]
            All the window sizes
        strides : List[int]
            The strides

        """
        self.features = []
        for function in functions:
            for key in signal_keys:
                for window in windows:
                    for stride in strides:
                        self.features.append(Feature(function, key, window, stride))


class FeatureCollection:
    """Collection of features to be calculated."""

    # TODO Add support for numpy functions without NumpyFuncWrapper

    def __init__(
        self, features_list: Union[List[Feature], List[MultipleFeatures]] = None
    ):
        """Create a FeatureCollection.

        Parameters
        ----------
        features_list : Union[List[Feature], List[MultipleFeatures]], optional
            Initial list of Features to add to collection, by default None
        """
        # The feature collection is a dict where the key is a tuple(str, int, int), the
        # tuple values correspond to (signal_key, window, stride)
        self._features_dict: Dict(tuple(str, int, int), List[Feature]) = {}
        # A list of all the features, holds the same references as the dict above but
        # is simply stored in another way
        self._features_list: List[Feature] = []
        if features_list:
            self.add(features_list)

    def _get_collection_key(self, feature: Feature):
        return (feature.key, feature.window, feature.stride)

    def _add_feature(self, feature: Feature):
        self._features_list.append(feature)

        key = self._get_collection_key(feature)
        if key in self._features_dict.keys():
            self._features_dict[key].append(feature)
        else:
            self._features_dict[key] = [feature]

    def add(self, features_list: Union[List[Feature], List[MultipleFeatures]]):
        """Add a list of features to the FeatureCollection.

        Parameters
        ----------
        features_list : Union[List[Feature], List[MultipleFeatures]]
            List of features to add.
        """
        for feature in features_list:
            if isinstance(feature, MultipleFeatures):
                self.add(feature.features)
            elif isinstance(feature, Feature):
                self._add_feature(feature)

    def calculate(self, signals: Union[List[pd.Series], pd.DataFrame]):
        """Calculate features on the passed singals.

        Parameters
        ----------
        signals : Union[List[pd.Series], pd.DataFrame]
            Dataframe or Series list with all the required signals for the feature
            calculation.

        Raises
        ------
        KeyError
            Raised when a needed key is not found in `signals`.
        """
        series_dict = dict()

        if isinstance(signals, pd.DataFrame):
            series_list = [signals[s] for s in signals.columns]
        else:
            series_list = signals

        for s in series_list:
            assert isinstance(s, pd.Series), "Error non pd.Series object passed"
            series_dict[s.name] = s.copy()

        # TODO add MultiProcessing
        # For all the operation on the same stridedRolling object
        for key in self._features_dict.keys():
            try:
                stroll = StridedRolling(series_dict[key[0]], key[1], key[2])
            except KeyError:
                raise KeyError("Key {} not found in series dict.".format(key[0]))

            for feature in self._features_dict[key]:
                if feature.output is None:
                    print(f"Feature calculation: {feature}")
                    df = stroll.apply_func(feature.function)
                    feature.output = df

    def get_results(self, merge_dfs=False) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Return the feature outputs.

        Parameters
        ----------
        merge_dfs : bool, optional
            Whether the results should be merged to a DataFrame wiht an outer merge
            , by default False

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            A DataFrame or List of DataFrames with the features in it.
        """
        # TODO maybe support merge asof as before?
        if merge_dfs:
            merged_df = pd.DataFrame()
            for feature in self._features_list:
                out_df = feature.output
                merged_df = pd.merge(
                    left=merged_df,
                    right=out_df,
                    how="outer",
                    left_index=True,
                    right_index=True,
                )
            return merged_df
        else:
            results = list(map(lambda feature: feature.output, self._features_list))
            return results

    def __repr__(self):
        """Representation string of FeatureCollection."""
        repr_string = f"{self.__class__.__name__}(\n"
        for feature in self._features_list:
            repr_string += f"\t{repr(feature)} \n"
        return repr_string + ')'
