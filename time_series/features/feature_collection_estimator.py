"""FeatureCollectionEstimator class for wrapped sklearn-compatible `FeatureCollection`.

See Also
--------
Example notebooks.

"""

from __future__ import annotations  # Make typing work for the enclosing class

__author__ = "Jeroen Van Der Donckt"

import pandas as pd

from typing import List, Union
from sklearn.base import BaseEstimator, TransformerMixin

from .feature_collection import FeatureCollection


class FeatureCollectionEstimator(BaseEstimator, TransformerMixin):
    """Sklearn-compatible estimator for extracting features using a `FeatureCollection`.

    This class is basically a wrapper around `FeatureCollection` and its `calculate`
    method, enabling sklearn compatibality, e.g. including an instance of this class
    in `sklearn.pipeline.Pipeline`.

    Concretely two changes were necessary to enable this sklearn-compatibility

    1. The parameters of the `FeatureCollection` its `calculate` method have been
        moved to the constructor
    2. The `FeatureCollection` its 'calculate' method is wrapped in this class its
        `transform` method.

    """

    def __init__(
        self,
        feature_collection: FeatureCollection,
        **feature_collection_calculate_kwargs
        # merge_dfs: Optional[bool] = False,
        # show_progress: Optional[bool] = False,
        # logging_file_path: Optional[Union[str, Path]] = None,
        # n_jobs: Optional[int] = None,
    ):
        """Create a FeatureCollectionEstimator.

        Note
        ----
        This constructor wraps a `FeatureCollection` instance and its paremeters of the
        `calculate` method.

        Parameters
        ---------
        feature_collection : FeatureCollection
            The `FeatureCollection` object that should be wrapped into this estimator.
        feature_collection_calculate_kwargs : Dict
            The keyword arguments used in the `FeatureCollection` its `calculate`
            method (i.e., show_progress, logging_file_path, n_jobs).
            Note that merge_dfs argument cannot be passed because this should always be
            True (in order to output a valid iterable after the `transform` method).
            See more (info on these arguments) in the documentation `FeatureCollection`
            itc `calculate` method.
        # show_progress: bool, optional
        #     If True, the progress will be shown with a progressbar, by default True.
        # logging_file_path: str, optional
        #     The file path where the logged messages are stored. If `None`, then no 
        #     logging `FileHandler` will be used and the logging messages are only pushed
        #     to stdout. Otherwise, a logging `FileHandler` will write the logged messages
        #     to the given file path.
        # n_jobs : int, optional
        #     The number of processes used for the feature calculation. If `None`, then
        #     the number returned by `os.cpu_count()` is used, by default None.
        #     If n_jobs is either 0 or 1, the code will be executed sequentially without
        #     creating a process pool. This is very useful when debugging, as the stack
        #     trace will be more comprehensible.

        Notes
        -----
        * If a `logging_file_path` is provided, the execution (time) statistics can be
          retrieved by calling `logger.get_function_duration_stats(logging_file_path)` 
          and `logger.get_key_duration_stats(logging_file_path)`.
          Be aware that the `logging_file_path` gets cleared before the logger pushes
          logged messages. Hence, one should use a separate logging file for the
          processing and the feature part of this library.
        * It is not possible to pass the `merge_dfs` argument in the 
          feature_collection_calculate_kwargs, because this should alwys be True
          (in order to output a valid iterable in the `transform` method)

        """
        self.feature_collection = feature_collection
        assert not 'merge_dfs' in feature_collection_calculate_kwargs.keys()
        self.calculate_kwargs = feature_collection_calculate_kwargs
        # self.show_progress = show_progress
        # self.logging_file_path = logging_file_path
        # self.n_jobs = n_jobs

    def fit(self, X=None, y=None) -> FeatureCollectionEstimator:
        """Fit function that is not needed for this estimator.

        Note
        ----
        This function does nothing and is here for compatibility reasons.

        Parameters
        ----------
        X : Any
            Unneeded.
        y : Any
            Unneeded.

        Returns
        -------
        FeatureCollectionEstimator
            The estimator instance itself is returned.
        """
        return self

    def transform(
        self,
        signals: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
    ) -> pd.DataFrame:
        """Calculate features on the signals.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]
            Dataframe or Series list with all the required signals for the feature
            calculation.
            Note that this parameter corresponds to the `signals` parameter of the
            `FeatureCollection` its `calculate` method.

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            A DataFrame or List of DataFrames with the features in it.

        """
        # TODO: can a list of dataframes be returned?? => NOPE
        return self.feature_collection.calculate(
            signals, merge_dfs=True, **self.calculate_kwargs
        )
