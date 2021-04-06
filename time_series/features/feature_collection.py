"""FeatureCollection class for collection and calculation of features."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

import pandas as pd

from typing import List, Union, Dict, Tuple, Iterator
from multiprocessing import Pool

from ..features.function_wrapper import NumpyFuncWrapper
from .strided_rolling import StridedRolling
from .feature import FeatureDescription, MultipleFeatureDescriptions


class FeatureCollection:
    """Collection of features to be calculated."""

    def __init__(
        self,
        feature_desc_list: Union[
            List[FeatureDescription], List[MultipleFeatureDescriptions]
        ] = None,
    ):
        """Create a FeatureCollection.

        Parameters
        ----------
        features_list : Union[List[Feature], List[MultipleFeatures]], optional
            Initial list of Features to add to collection, by default None

        """
        # The feature collection is a dict where the key is a tuple(str, int, int), the
        # tuple values correspond to (signal_key, window, stride)
        self._feature_desc_dict: Dict[
            Tuple[str, int, int], List[FeatureDescription]
        ] = {}
        # A list of all the features, holds the same references as the dict above but
        # is simply stored in another way
        self._feature_desc_list: List[FeatureDescription] = []
        if feature_desc_list:
            self.add(feature_desc_list)

    @staticmethod
    def _get_collection_key(feature: FeatureDescription):
        return feature.key, feature.window, feature.stride

    def _add_feature(self, feature: FeatureDescription):
        self._feature_desc_list.append(feature)

        key = self._get_collection_key(feature)
        if key in self._feature_desc_dict.keys():
            self._feature_desc_dict[key].append(feature)
        else:
            self._feature_desc_dict[key] = [feature]

    def add(
        self,
        features_list: Union[
            List[FeatureDescription], List[MultipleFeatureDescriptions]
        ],
    ):
        """Add a list of FetaureDescription to the FeatureCollection.

        Parameters
        ----------
        features_list : Union[List[Feature], List[MultipleFeatures]]
            List of features to add.

        """
        for feature in features_list:
            if isinstance(feature, MultipleFeatureDescriptions):
                self.add(feature.feature_descriptions)
            elif isinstance(feature, FeatureDescription):
                self._add_feature(feature)

    @staticmethod
    def _executor(stroll: StridedRolling, function: NumpyFuncWrapper):
        return stroll.apply_func(function)

    def _stroll_feature_generator(
        self, series_dict: Dict[str, pd.Series]
    ) -> Iterator[Tuple[StridedRolling, NumpyFuncWrapper]]:
        # We could also make the StridedRolling creation multithreaded
        # Another possible option to speed up this creations by making this lazy
        # and only creating it upon calling.
        for signal_key, win, stride in self._feature_desc_dict.keys():
            try:
                stroll = StridedRolling(series_dict[signal_key], win, stride)
            except KeyError:
                raise KeyError(f"Key {signal_key} not found in series dict.")

            for feature in self._feature_desc_dict[(signal_key, win, stride)]:
                yield (stroll, feature.function)

    def calculate(
        self,
        signals: Union[List[pd.Series], pd.DataFrame],
        merge_dfs=False,
        njobs=None,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Calculate features on the passed signals.

        Parameters
        ----------
        signals : Union[List[pd.Series], pd.DataFrame]
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

        if isinstance(signals, pd.DataFrame):
            series_list = [signals[s] for s in signals.columns]
        else:
            series_list = signals

        for s in series_list:
            assert isinstance(s, pd.Series), "Error non pd.Series object passed"
            series_dict[s.name] = s

        calculated_feature_list: List[pd.DataFrame] = []

        with Pool(processes=njobs) as pool:
            calculated_feature_list.extend(
                pool.starmap(
                    self._executor, self._stroll_feature_generator(series_dict)
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

    def __repr__(self):
        """Representation string of FeatureCollection."""
        repr_string = f"{self.__class__.__name__}(\n"
        for feature in self._feature_desc_list:
            repr_string += f"\t{repr(feature)} \n"
        return repr_string + ")"
