"""FeatureCollection class for collection and calculation of features.

See Also
--------
Example notebooks and model serialization documentation.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import dill
import pandas as pd
from pathos.multiprocessing import ProcessPool

from ..features.function_wrapper import NumpyFuncWrapper
from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .strided_rolling import StridedRolling


class FeatureCollection:
    """Collection of features to be calculated."""

    def __init__(
        self,
        feature_desc_list: Union[
            List[FeatureDescriptor], List[MultipleFeatureDescriptors]
        ] = None,
    ):
        """Create a FeatureCollection.

        Parameters
        ----------
        feature_desc_list : Union[List[Feature], List[MultipleFeatures]], optional
            Initial list of Features to add to collection, by default None

        """
        # The feature collection is a dict where the key is a tuple(str, int, int), the
        # tuple values correspond to (signal_key, window, stride)
        self._feature_desc_dict: Dict[
            Tuple[str, int, int], List[FeatureDescriptor]
        ] = {}
        # A list of all the features, holds the same references as the dict above but
        # is simply stored in another way
        self._feature_desc_list: List[FeatureDescriptor] = []
        if feature_desc_list:
            self.add(feature_desc_list)

    @staticmethod
    def _get_collection_key(feature: FeatureDescriptor):
        return feature.key, feature.window, feature.stride

    def _add_feature(self, feature: FeatureDescriptor):
        """Add a `FeatureDescriptor` instance to the collection.

        Parameters
        ----------
        feature : FeatureDescriptor
            The featuredescriptor that will be added.

        """
        self._feature_desc_list.append(feature)

        key = self._get_collection_key(feature)
        if key in self._feature_desc_dict.keys():
            self._feature_desc_dict[key].append(feature)
        else:
            self._feature_desc_dict[key] = [feature]

    def add(
        self,
        features_list: Union[
            List[FeatureDescriptor],
            List[MultipleFeatureDescriptors],
            List[FeatureCollection],
        ],
    ):
        """Add a list of FeatureDescription to the FeatureCollection.

        Todo
        ----
        Type hint of `feature_list` is not totally correct.

        Parameters
        ----------
        features_list : Union[List[FeatureDescriptor], List[MultipleFeatureDescriptors], List[FeatureCollection]],
            List of feature(containers) which features will be added.

        Raises
        ------
        TypeError
            Raised when an item within `features_list` is not an instance of
            [`MultipleFeatureDescriptors`, `FeatureDescriptors`, `FeatureCollection`].

        """
        for feature in features_list:
            if isinstance(feature, MultipleFeatureDescriptors):
                self.add(feature.feature_descriptions)
            elif isinstance(feature, FeatureDescriptor):
                self._add_feature(feature)
            elif isinstance(feature, FeatureCollection):
                self.add(feature._feature_desc_list)
            else:
                raise TypeError(f"type: {type(feature)} is not supported")

    @staticmethod
    def _executor(stroll: StridedRolling, function: NumpyFuncWrapper):
        return stroll.apply_func(function)

    def _stroll_generator(
        self, series_dict: Dict[str, pd.Series]
    ) -> Iterator[StridedRolling]:
        # We could also make the StridedRolling creation multithreaded
        # Another possible option to speed up this creations by making this lazy
        # and only creating it upon calling.
        for feature in self._feature_desc_list:
            try:
                stroll = StridedRolling(
                    series_dict[feature.key], feature.window, feature.stride
                )
            except KeyError:
                raise KeyError(f"Key {feature.key} not found in series dict.")
            yield stroll

    def calculate(
        self,
        signals: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        merge_dfs=False,
        njobs=None,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Calculate features on the passed signals.

        Note
        ----
        The column-names of the signals represent the signal-keys.

        Parameters
        ----------
        signals : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]
            Dataframe or Series list with all the required signals for the feature
            calculation.
        merge_dfs : bool, optional
            Whether the results should be merged to a DataFrame with an outer merge,
            by default False
        njobs : int, optional
            The number of processes used for the feature calculation. If `None`, then
            the number returned by `os.cpu_count()` is used, by default None.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            A DataFrame or List of DataFrames with the features in it.

        Raises
        ------
        KeyError
            Raised when a required key is not found in `signals`.

        """
        series_dict = dict()
        series_list = []

        if not isinstance(signals, list):
            signals = [signals]

        for s in signals:
            if isinstance(s, pd.DataFrame):
                series_list += [s[c] for c in s.columns]
            elif isinstance(s, pd.Series):
                series_list.append(s)
            else:
                raise TypeError("Non pd.Series or pd.DataFrame object passed.")

        for s in series_list:
            series_dict[s.name] = s

        calculated_feature_list: List[pd.DataFrame] = []

        # https://pathos.readthedocs.io/en/latest/pathos.html#usage
        # nodes = number (and potentially description) of workers
        # ncpus - number of worker processors servers
        with ProcessPool(nodes=njobs) as pool:
            calculated_feature_list.extend(
                pool.map(
                    self._executor,
                    self._stroll_generator(series_dict),
                    (x.function for x in self._feature_desc_list),
                )
            )

        if merge_dfs:
            df_merged = pd.DataFrame()
            for calculated_feature in calculated_feature_list:
                df_merged = pd.merge(
                    left=df_merged,
                    right=calculated_feature,
                    how="outer",
                    left_index=True,
                    right_index=True,
                )
            return df_merged
        else:
            return calculated_feature_list

    def serialize(self, file_path: Union[str, Path]):
        """Serialize this `FeatureCollection` instance.

        Note
        ----
        As we use `dill` to serialize the files, we can also serialize functions which
        are defined in the local scope, like lambdas.

        Parameters
        ----------
        file_path : Union[str, Path]
            The path where the `FeatureCollection` will be serialized.

        See Also
        --------
        https://github.com/uqfoundation/dill

        """
        with open(file_path, "wb") as f:
            dill.dump(self, f, recurse=True)

    def __repr__(self) -> str:
        """Representation string of a Featurecollection."""
        signals = sorted(set(k[0] for k in self._feature_desc_dict.keys()))
        output_str = ''
        for signal in signals:
            output_str += f"{signal}: ("
            keys = (x for x in self._feature_desc_dict.keys() if x[0] == signal)
            for _, win_size, stride in keys:
                output_str += f'\n\twin: {str(win_size):<6}, stride: {str(stride)}: ['
                for feat_desc in self._feature_desc_dict[signal, win_size, stride]:
                    output_str += f"\n\t\t{feat_desc._func_str()},"
                output_str += '\n\t]'
            output_str += "\n)\n"
        return output_str
