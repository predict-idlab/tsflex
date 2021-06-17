"""SKFeatureCollection class for wrapped sklearn-compatible `FeatureCollection`.

.. todo::
    * **BaseEstimator support**: it is not really useful right now to support this.
    As for example sklearn GridSearchCV requires X and y to have the same length,
    but `FeatureCollection`.`calculate` (almost always) transforms the length of X in
    your pipeline.<br>
    _Possible solution_: look into sklearn-contrib imblearn how they handle this

See Also
--------
Example notebooks.

"""

from __future__ import annotations  # Make typing work for the enclosing class

__author__ = "Jeroen Van Der Donckt"

import pandas as pd

from typing import List, Optional, Union, Any
from pathlib import Path
from sklearn.base import TransformerMixin

from .feature_collection import FeatureCollection
from .feature import FeatureDescriptor, MultipleFeatureDescriptors


class SKFeatureCollection(TransformerMixin):
    """Sklearn-compatible transformer for extracting features using `FeatureCollection`.

    This class is basically a wrapper around `FeatureCollection` and its `calculate`
    method, enabling sklearn compatibility, e.g. including an instance of this class
    in `sklearn.pipeline.Pipeline`. The constructor wraps the `FeatureCollection` its
    constructor arguments and the  relevant parameters of its `calculate` method.

    Parameters
    ----------
    feature_descriptors : Union[FeatureDescriptor, MultipleFeatureDescriptors, List[Union[FeatureDescriptor, MultipleFeatureDescriptors]]]
        Features to include in the collection.
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

    Notes
    -----
    * If a `logging_file_path` is provided, the execution (time) info can be
      retrieved by calling `logger.get_feature_logs(logging_file_path)`. <br>
      Be aware that the `logging_file_path` gets cleared before the logger pushes
      logged messages. Hence, one should use a separate logging file for each
      constructed processing and feature instance with this library.
    * It is not possible to pass the `merge_dfs` argument from the `calculate`
      method, because this should always be True (in order to output a valid
      iterable in the `transform` method)\n
    The following three changes were necessary to enable this sklearn-compatibility: \n
    1. The parameters of the `FeatureCollection` its constructor are logged in this
       class.
    2. The relevant parameters of the `FeatureCollection` its `calculate` method have
       been moved to the constructor and are also logged.
    3. The `FeatureCollection` its `calculate` method is wrapped in this class its
       `transform` method,

    """

    def __init__(
        self,
        feature_descriptors: Union[FeatureDescriptor, MultipleFeatureDescriptors, List[
            Union[FeatureDescriptor, MultipleFeatureDescriptors]]],
        logging_file_path: Optional[Union[str, Path]] = None,
        n_jobs: Optional[int] = None,
    ):
        self.feature_descriptors: Any = feature_descriptors
        self.logging_file_path = logging_file_path
        self.n_jobs = n_jobs

    def fit(self, X=None, y=None) -> SKFeatureCollection:
        """Fit function that is not needed for this transformer.

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
        SKFeatureCollection
            The transformer instance itself is returned.

        """
        return self

    def transform(
        self,
        X: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
    ) -> pd.DataFrame:
        """Calculate features on the data.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            Dataframe or Series or list thereof, with all the required data for the
            feature calculation.

            **Remark**: each Series/DataFrame must have a `pd.DatetimeIndex`. \n
            Also note that this parameter corresponds to the `data` parameter of the
            `FeatureCollection` its `calculate` method.

        Returns
        -------
        pd.DataFrame
            A DataFrame, containing the calculated features.

        """
        feature_collection = FeatureCollection(self.feature_descriptors)
        return feature_collection.calculate(
            data=X,
            return_df=True,
            show_progress=False,
            logging_file_path=self.logging_file_path,
            n_jobs=self.n_jobs,
        )

    def __repr__(self) -> str:
        """Representation string of a SKFeatureCollection."""
        return repr(FeatureCollection(self.feature_descriptors))
