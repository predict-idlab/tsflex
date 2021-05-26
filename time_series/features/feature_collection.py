"""FeatureCollection class for collection and calculation of features.

See Also
--------
Example notebooks and model serialization documentation.

"""

from __future__ import annotations

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

import dill
import logging
import warnings
import pandas as pd

from pathos.multiprocessing import ProcessPool
from tqdm.auto import tqdm
from pathlib import Path

from typing import Dict, Iterator, List, Optional, Tuple, Union

from ..features.function_wrapper import NumpyFuncWrapper
from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .strided_rolling import StridedRolling
from .logger import logger


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
    def _executor(t: Tuple[StridedRolling, NumpyFuncWrapper]):
        stroll = t[0]
        function = t[1]
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
                yield stroll, feature.function

    def calculate(
        self,
        signals: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        merge_dfs=False,
        njobs: int = None,
        logging_file_path: Optional[Union[str, Path]] = None,
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
        logging_file_path: str, optional
            The file path where the logged messages are stored. If `None`, then no 
            logging `FileHandler` will be used and the logging messages are only pushed
            to stdout. Otherwise, a logging `FileHandler` will write the logged messages
            to the given file path.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            A DataFrame or List of DataFrames with the features in it.

        Note
        ----
        If a `logging_file_path` is provided, the execution (time) statistics can be
        retrieved by calling `logger.get_function_duration_stats(logging_file_path)` and
        `logger.get_key_duration_stats(logging_file_path)`.

        Raises
        ------
        KeyError
            Raised when a required key is not found in `signals`.

        """

        # Delete other logging handlers
        if len(logger.handlers) > 1:
            logger.handlers = [h for h in logger.handlers if type(h) == logging.StreamHandler]
        assert len(logger.handlers) == 1, 'Multiple logging StreamHandlers present!!'

        if logging_file_path:
            if not isinstance(logging_file_path, Path):
                logging_file_path = Path(logging_file_path)
            if logging_file_path.exists():
                warnings.warn(
                    f"Logging file ({logging_file_path}) already exists. This file will be overwritten!"
                )
                # Clear the file
                #  -> because same FileHandler is used when calling this method twice
                open(logging_file_path, 'w').close()
            f_handler = logging.FileHandler(logging_file_path, mode="w")
            f_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            f_handler.setLevel(logging.INFO)
            logger.addHandler(f_handler)

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
            results = pool.uimap(
                self._executor, self._stroll_feature_generator(series_dict)
            )
            for f in tqdm(results, total=len(self._feature_desc_list)):
                calculated_feature_list.append(f)
            # Close & join because: https://github.com/uqfoundation/pathos/issues/131
            pool.close()
            pool.join()
            # Clear because: https://github.com/uqfoundation/pathos/issues/111
            pool.clear()

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
        output_str = ""
        for signal in signals:
            output_str += f"{signal}: ("
            keys = (x for x in self._feature_desc_dict.keys() if x[0] == signal)
            for _, win_size, stride in keys:
                output_str += f"\n\twin: {str(win_size):<6}, stride: {str(stride)}: ["
                for feat_desc in self._feature_desc_dict[signal, win_size, stride]:
                    output_str += f"\n\t\t{feat_desc._func_str()},"
                output_str += "\n\t]"
            output_str += "\n)\n"
        return output_str
