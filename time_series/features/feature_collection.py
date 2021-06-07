"""FeatureCollection class for collection and calculation of time-series features.

See Also
--------
Example notebooks and model serialization documentation.

"""

from __future__ import annotations  # Make typing work for the enclosing class

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
from ..utils.data import series_dict_to_df, to_series_list
from ..utils.timedelta import timedelta_to_str
from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .strided_rolling import StridedRolling
from .logger import logger


class FeatureCollection:
    """Collection of features to be calculated."""

    def __init__(
        self,
        feature_descriptors: Optional[
            Union[
                FeatureDescriptor, MultipleFeatureDescriptors,
                List[Union[FeatureDescriptor, MultipleFeatureDescriptors]]
            ]
        ] = None,
    ):
        """Create a FeatureCollection.

        Parameters
        ----------
        feature_descriptors : Union[FeatureDescriptor, MultipleFeatureDescriptors, List[Union[FeatureDescriptor, MultipleFeatureDescriptors]]], optional
            Initial (list of) feature(s) to add to collection, by default None

        """
        # The feature collection is a dict where the key is a tuple(str, int, int), the
        # tuple values correspond to (series_key(s), window, stride)
        self._feature_desc_dict: Dict[
            Tuple[Tuple[str], Union[int, pd.Timedelta], Union[int, pd.Timedelta]],
            List[FeatureDescriptor],
        ] = {}
        # A list of all the features, holds the same references as the dict above but
        # is simply stored in another way
        self._feature_desc_list: List[FeatureDescriptor] = []
        if feature_descriptors:
            self.add(feature_descriptors)

    @staticmethod
    def _get_collection_key(feature: FeatureDescriptor):
        # Note `window` & `stride` properties can either be a pd.Timedelta or an int
        return feature.key, feature.window, feature.stride

    def _add_feature(self, feature: FeatureDescriptor):
        """Add a `FeatureDescriptor` instance to the collection.

        Parameters
        ----------
        feature : FeatureDescriptor
            The feature that will be added to this feature collection.

        """
        self._feature_desc_list.append(feature)

        key = self._get_collection_key(feature)
        if key in self._feature_desc_dict.keys():
            self._feature_desc_dict[key].append(feature)
        else:
            self._feature_desc_dict[key] = [feature]

    def add(
        self,
        features: Union[
            FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection,
            List[
                Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection]
            ]
        ],
    ):
        """Add feature(s) to the FeatureCollection.

        Parameters
        ----------
        features : Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection, List[Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection]]]
            Feature(s) (containers) which contained features will be added.

        Raises
        ------
        TypeError
            Raised when an item within `features` is not an instance of
            [`MultipleFeatureDescriptors`, `FeatureDescriptors`, `FeatureCollection`].

        """
        if not isinstance(features, list):
            features = [features]
        for feature in features:
            if isinstance(feature, MultipleFeatureDescriptors):
                self.add(feature.feature_descriptions)
            elif isinstance(feature, FeatureDescriptor):
                self._add_feature(feature)
            elif isinstance(feature, FeatureCollection):
                self.add(feature._feature_desc_list)
            else:
                raise TypeError(f"type: {type(feature)} is not supported")

    @staticmethod
    def _executor(t: Tuple[StridedRolling, NumpyFuncWrapper, bool]):
        stroll = t[0]
        function = t[1]
        single_series_func = t[2]
        return stroll.apply_func(function, single_series_func)

    def _stroll_feature_generator(
        self, series_dict: Dict[str, pd.Series]
    ) -> Iterator[Tuple[StridedRolling, NumpyFuncWrapper]]:
        # We could also make the StridedRolling creation multithreaded
        # Another possible option to speed up this creations by making this lazy
        # and only creating it upon calling.
        def get_feature_df(feature_key: Tuple[str]) -> Union[pd.Series, pd.DataFrame]:
            """Get the data for the feature.
            
            Returns
            * `pd.Series` for a *single input-series function*
            * `pd.Dataframe` for a *multiple input-series function*
              (this dataframe contains the merged series).
            """
            if len(feature_key) == 1:
                # Very efficient => return just the reference to the single series
                return series_dict[feature_key[0]]
            # Otherwise create efficiently a dataframe for the multiple series
            return series_dict_to_df({name: series_dict[name] for name in feature_key})

        for key, win, stride in self._feature_desc_dict.keys():
            try:
                stroll = StridedRolling(get_feature_df(key), win, stride)
            except KeyError:
                raise KeyError(f"Key {key} not found in series dict.")

            for feature in self._feature_desc_dict[(key, win, stride)]:
                yield stroll, feature.function, feature.is_single_series_func()

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        merge_dfs: Optional[bool] = False,
        show_progress: Optional[bool] = False,
        logging_file_path: Optional[Union[str, Path]] = None,
        n_jobs: Optional[int] = None,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Calculate features on the passed data.

        Note
        -----
        The (column-)names of the series in `data` represent the names in the keys.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            Dataframe or Series or list thereof, with all the required data for the
            feature calculation. \n
            **Remark**: each Series/DataFrame must have a `pd.DatetimeIndex`.
        merge_dfs : bool, optional
            Whether the results should be merged to a DataFrame with an outer merge,
            by default False
        show_progress: bool, optional
            If True, the progress will be shown with a progressbar, by default False.
        logging_file_path : Union[str, Path], optional
            The file path where the logged messages are stored. If `None`, then no
            logging `FileHandler` will be used and the logging messages are only pushed
            to stdout. Otherwise, a logging `FileHandler` will write the logged messages
            to the given file path.
        n_jobs : int, optional
            The number of processes used for the feature calculation. If `None`, then
            the number returned by `os.cpu_count()` is used, by default None. \n
            If n_jobs is either 0 or 1, the code will be executed sequentially without
            creating a process pool. This is very useful when debugging, as the stack
            trace will be more comprehensible.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            A DataFrame or List of DataFrames with the features in it.

        Note
        ----
        If a `logging_file_path` is provided, the execution (time) statistics can be
        retrieved by calling `logger.get_function_duration_stats(logging_file_path)`
        and `logger.get_key_duration_stats(logging_file_path)`. <br>
        Be aware that the `logging_file_path` gets cleared before the logger pushes
        logged messages. Hence, one should use a separate logging file for each
        constructed processing and feature instance with this library.

        Raises
        ------
        KeyError
            Raised when a required key is not found in `data`.

        """
        # Delete other logging handlers
        if len(logger.handlers) > 1:
            logger.handlers = [
                h for h in logger.handlers if type(h) == logging.StreamHandler
            ]
        assert len(logger.handlers) == 1, "Multiple logging StreamHandlers present!!"

        if logging_file_path:
            if not isinstance(logging_file_path, Path):
                logging_file_path = Path(logging_file_path)
            if logging_file_path.exists():
                warnings.warn(
                    f"Logging file ({logging_file_path}) already exists. This file will be overwritten!"
                )
                # Clear the file
                #  -> because same FileHandler is used when calling this method twice
                open(logging_file_path, "w").close()
            f_handler = logging.FileHandler(logging_file_path, mode="w")
            f_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            f_handler.setLevel(logging.INFO)
            logger.addHandler(f_handler)

        # Convert the data to a series_dict
        series_list = to_series_list(data)
        series_dict : Dict[str, pd.Series] = {}
        for s in series_list:
            series_dict[s.name] = s

        calculated_feature_list: List[pd.DataFrame] = []

        # https://pathos.readthedocs.io/en/latest/pathos.html#usage
        if n_jobs in [0, 1]:
            # print('Executing feature extraction sequentially')
            for stroll, func, single_series_func in self._stroll_feature_generator(
                series_dict
            ):
                calculated_feature_list.append(
                    stroll.apply_func(func, single_series_func)
                )
        else:
            with ProcessPool(nodes=n_jobs, source=True) as pool:
                results = pool.uimap(
                    self._executor, self._stroll_feature_generator(series_dict)
                )
                if show_progress:
                    results = tqdm(results, total=len(self._feature_desc_list))
                for f in results:
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
        """Representation string of a FeatureCollection."""
        feature_keys = sorted(set(k[0] for k in self._feature_desc_dict.keys()))
        output_str = ""
        for feature_key in feature_keys:
            output_str += f"{feature_key}: ("
            keys = (x for x in self._feature_desc_dict.keys() if x[0] == feature_key)
            for _, win_size, stride in keys:
                output_str += f"\n\twin: "
                win_str, stride_str = win_size, stride
                if isinstance(win_str, pd.Timedelta):
                    win_str = timedelta_to_str(win_str)
                else:
                    win_str = f"{win_str} samples"
                if isinstance(stride_str, pd.Timedelta):
                    stride_str = timedelta_to_str(stride_str)
                else:
                    stride_str = f"{stride_str} samples"
                output_str += f"{str(win_str):<6}, stride: {str(stride_str)}: ["
                for feat_desc in self._feature_desc_dict[feature_key, win_size, stride]:
                    output_str += f"\n\t\t{feat_desc._func_str()},"
                output_str += "\n\t]"
            output_str += "\n)\n"
        return output_str
