"""FeatureCollection class for bookkeeping and calculation of time-series features.

See Also
--------
Example notebooks and model serialization documentation.

"""

from __future__ import annotations  # Make typing work for the enclosing class

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

import logging
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import dill
import pandas as pd
from pathos.multiprocessing import ProcessPool
from tqdm.auto import tqdm

from .feature import FeatureDescriptor, MultipleFeatureDescriptors
from .logger import logger
from .strided_rolling import StridedRolling
from ..features.function_wrapper import NumpyFuncWrapper
from ..utils.data import to_list, to_series_list, flatten
from ..utils.timedelta import timedelta_to_str


class FeatureCollection:
    """Create a FeatureCollection.

    Parameters
    ----------
    feature_descriptors : Union[FeatureDescriptor, MultipleFeatureDescriptors, List[Union[FeatureDescriptor, MultipleFeatureDescriptors]]], optional
        Initial (list of) feature(s) to add to collection, by default None

    """

    def __init__(
        self,
        feature_descriptors: Optional[
            Union[
                FeatureDescriptor, MultipleFeatureDescriptors,
                List[Union[FeatureDescriptor, MultipleFeatureDescriptors]]
            ]
        ] = None,
    ):
        # The feature collection is a dict with keys of type:
        #   tuple(tuple(str), float OR pd.timedelta, float OR pd.timedelta)
        # The outer tuple's values correspond to (series_key(s), window, stride)
        self._feature_desc_dict: Dict[
            Tuple[Tuple[str], pd.Timedelta, pd.Timedelta], List[FeatureDescriptor]
        ] = {}

        if feature_descriptors:
            self.add(feature_descriptors)

    def get_required_series(self) -> List[str]:
        """Return all required series names for this feature collection.

        Return the list of series names that are required in order to calculate all the
        features (defined by the `FeatureDescriptor` objects) of this feature
        collection.

        Returns
        -------
        List[str]
            List of all the required series names.

        """
        return list(
            set(
                flatten(
                    [fr_key[0] for fr_key in self._feature_desc_dict.keys()]
                )
            )
        )

    @staticmethod
    def _get_collection_key(feature: FeatureDescriptor)\
            -> Tuple[tuple, pd.Timedelta, pd.Timedelta]:
        # Note: `window` & `stride` properties can either be a pd.Timedelta or an int
        return feature.series_name, feature.window, feature.stride

    def _add_feature(self, feature: FeatureDescriptor):
        """Add a `FeatureDescriptor` instance to the collection.

        Parameters
        ----------
        feature : FeatureDescriptor
            The feature that will be added to this feature collection.

        """
        series_win_stride_key = self._get_collection_key(feature)
        if series_win_stride_key in self._feature_desc_dict.keys():
            self._feature_desc_dict[series_win_stride_key].append(feature)
        else:
            self._feature_desc_dict[series_win_stride_key] = [feature]

    def add(
        self,
        features: Union[
            FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection, List[
                Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection]
            ]
        ],
    ):
        """Add feature(s) to the FeatureCollection.

        Parameters
        ----------
        features : Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection, List[Union[FeatureDescriptor, MultipleFeatureDescriptors, FeatureCollection]]]
            Feature(s) (containers) whose contained features will be added.

        Raises
        ------
        TypeError
            Raised when an item within `features` is not an instance of
            [`MultipleFeatureDescriptors`, `FeatureDescriptors`, `FeatureCollection`].

        """
        # Convert to list if necessary
        features = to_list(features)

        for feature in features:
            if isinstance(feature, MultipleFeatureDescriptors):
                self.add(feature.feature_descriptions)
            elif isinstance(feature, FeatureDescriptor):
                self._add_feature(feature)
            elif isinstance(feature, FeatureCollection):
                # List needs to be flattened
                self.add(list(flatten(feature._feature_desc_dict.values())))
            else:
                raise TypeError(f"type: {type(feature)} is not supported - {feature}")

    @staticmethod
    def _executor(t: Tuple[StridedRolling, NumpyFuncWrapper]):
        stroll: StridedRolling = t[0]
        function: NumpyFuncWrapper = t[1]
        return stroll.apply_func(function)

    def _stroll_feature_generator(
        self, series_dict: Dict[str, pd.Series], window_idx: str, approve_sparsity: bool
    ) -> Iterator[Tuple[StridedRolling, NumpyFuncWrapper]]:
        # --- Future work ---
        # We could also make the StridedRolling creation multithreaded
        # Very low priority because the STROLL __init__ is rather efficient!
        for key, win, stride in self._feature_desc_dict.keys():
            try:
                stroll = StridedRolling(
                    data=[series_dict[k] for k in key],
                    window=win,
                    stride=stride,
                    window_idx=window_idx,
                    approve_sparsity=approve_sparsity,
                )
            except KeyError:
                raise KeyError(f"Key {key} not found in series dict.")

            for feature in self._feature_desc_dict[(key, win, stride)]:
                yield stroll, feature.function

    def calculate(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        return_df: Optional[bool] = False,
        window_idx: Optional[str] = 'end',
        approve_sparsity: Optional[bool] = False,
        show_progress: Optional[bool] = False,
        logging_file_path: Optional[Union[str, Path]] = None,
        n_jobs: Optional[int] = None,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Calculate features on the passed data.

        Notes
        ------
        * The (column-)names of the series in `data` represent the names in the keys.
        * If a `logging_file_path` is provided, the execution (time) info can be
          retrieved by calling `logger.get_feature_logs(logging_file_path)`.
          Be aware that the `logging_file_path` gets cleared before the logger pushes
          logged messages. Hence, one should use a separate logging file for each
          constructed processing and feature instance with this library.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            Dataframe or Series or list thereof, with all the required data for the
            feature calculation. \n
            **Remark**: each Series / DataFrame must have a `pd.DatetimeIndex`.
            **Remark**: we assume that each name / column is unique.
        return_df : bool, optional
            Whether the output needs to be a dataframe list or a DataFrame, by default 
            False.
            If `True` the output dataframes will be merged to a DataFrame with an outer
            merge.
        window_idx : str, optional
            The window's index position which will be used as index for the
            feature_window aggregation. Must be either of: ['begin', 'middle', 'end'],
            by default 'end'. All features in this collection will use the same
            window_idx.
        approve_sparsity: bool, optional
            Bool indicating whether the user acknowledges that there may be sparsity 
            (i.e., irregularly sampled data), by default False.
            If False and sparsity is observed, a warning is raised.
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

            .. tip::
                * It takes on avg. _300ms_ to schedule everything with
                  multiprocessing. So if your feature extraction code runs faster than
                  ~1.5s, it might not be worth it to parallelize the process
                  (and thus better set the `n_jobs` to 0-1).
                * This method its memory peaks are significantly lower when executed
                  sequentially. Set the `n_jobs` to 0-1 if this matters.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            The calculated features.

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
                    f"Logging file ({logging_file_path}) already exists. "
                    f"This file will be overwritten!"
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
        series_dict: Dict[str, pd.Series] = {}
        for s in to_series_list(data):
            # Assert the assumptions we make!
            assert isinstance(s.index, pd.DatetimeIndex)
            assert s.index.is_monotonic_increasing

            if s.name in self.get_required_series():
                series_dict[str(s.name)] = s

        calculated_feature_list: List[pd.DataFrame] = []

        if n_jobs in [0, 1]:
            # print('Executing feature extraction sequentially')
            for stroll, func in self._stroll_feature_generator(
                series_dict, window_idx, approve_sparsity
            ):
                calculated_feature_list.append(stroll.apply_func(func))
        else:
            # https://pathos.readthedocs.io/en/latest/pathos.html#usage
            with ProcessPool(nodes=n_jobs, source=True) as pool:
                results = pool.uimap(
                    self._executor,
                    self._stroll_feature_generator(
                        series_dict, window_idx, approve_sparsity
                    )
                )
                if show_progress:
                    results = tqdm(results, total=len(self._feature_desc_dict))
                for f in results:
                    calculated_feature_list.append(f)
                # Close & join - see: https://github.com/uqfoundation/pathos/issues/131
                pool.close()
                pool.join()
                # Clear because: https://github.com/uqfoundation/pathos/issues/111
                pool.clear()

        if return_df:
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

        Parameters
        ----------
        file_path : Union[str, Path]
            The path where the `FeatureCollection` will be serialized.

        Notes
        -----
        * As we use [Dill](https://github.com/uqfoundation/dill){:target="_blank"} to
          serialize the files, we can **also serialize functions which are defined in
          the local scope, like lambdas.**

        """
        with open(file_path, "wb") as f:
            dill.dump(self, f, recurse=True)

    def __repr__(self) -> str:
        """Representation string of a FeatureCollection."""
        feature_keys = sorted(set(k[0] for k in self._feature_desc_dict.keys()))
        output_str = ""
        for feature_key in feature_keys:
            output_str += f"{'|'.join(feature_key)}: ("
            keys = (x for x in self._feature_desc_dict.keys() if x[0] == feature_key)
            for _, win_size, stride in keys:
                output_str += f"\n\twin: "
                win_str = timedelta_to_str(win_size)
                stride_str = timedelta_to_str(stride)
                output_str += f"{str(win_str):<6}, stride: {str(stride_str)}: ["
                for feat_desc in self._feature_desc_dict[feature_key, win_size, stride]:
                    output_str += f"\n\t\t{feat_desc._func_str},"
                output_str += "\n\t]"
            output_str += "\n)\n"
        return output_str
