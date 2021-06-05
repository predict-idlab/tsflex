"""SKSeriesPipeline class for wrapped sklearn-compatible `SeriesPipeline`.

See Also
--------
Example notebooks.

"""

from __future__ import annotations  # Make typing work for the enclosing class

__author__ = "Jeroen Van Der Donckt"

import pandas as pd

from typing import List, Optional, Union
from pathlib import Path
from sklearn.base import TransformerMixin

from .series_processor import SeriesProcessor
from .series_pipeline import SeriesPipeline


## Future work
# * BaseEstimator support: it is not really useful right now to support this.
#   As for example sklearn GridSearchCV requires X and y to have the same length,
#   but SeriesPipeline (sometimes) transforms the length of X in your pipeline.
#   => Possible solution; look into sklearn-contrib imblearn how they handle this


class SKSeriesPipeline(TransformerMixin):
    """Sklearn-compatible transformer for processing signals using a `SeriesPipeline`.

    This class is basically a wrapper around `SeriesPipeline` and its `process`
    method, enabling sklearn compatibality, e.g. including an instance of this class
    in `sklearn.pipeline.Pipeline`.

    Concretely three changes were necessary to enable this sklearn-compatibility

    1. The parameters of the `SeriesPipeline` its constructor are logged in this
       class.
    2. The relevant parameters of the `SeriesPipeline` its `process` method have
       been moved to the constructor and are also logged.
    3. The `SeriesPipeline` its `process` method is wrapped in this class its
       `transform` method,

    """

    def __init__(
        self,
        processors: List[SeriesProcessor],
        return_all_signals: Optional[bool] = True,
        drop_keys: Optional[List[str]] = [],
        logging_file_path: Optional[Union[str, Path]] = None,
    ):
        """Create a `SKSeriesPipeline`, which is a sklearn-compatible transformer.

        This constructor wraps the `SeriesPipeline` constructor arguments and the
        relevant paremeters of its `process` method.

        Parameters
        ----------
        processors : List[SeriesProcessor]
            List of `SeriesProcessor` objects that will be applied sequentially to the
            given signals. The processing steps will be executed in the  same order as
            passed with this list.
        return_all_signals : bool, optional
            Whether the output needs to return all the signals, by default True.
            If `True` the output will contain all signals that were passed to this
            method. If `False` the output will contain just the required signals (see
            `get_all_required_signals`).
        return_df : bool, optional
            Whether the output needs to be a series dict or a DataFrame, default True.
            If `True` the output series will be combined to a DataFrame with an outer
            merge.
        drop_keys : List[str], optional
            Which keys should be dropped when returning the output, by default [].
        logging_file_path : Union[str, Path], optional
            The file path where the logged messages are stored, by default None.
            If `None`, then no logging `FileHandler` will be used and the logging
            messages are only pushed to stdout. Otherwise, a logging `FileHandler` will
            write the logged messages to the given file path.

        Notes
        -----
        *  If a `logging_file_path` is provided, the execution (time) statistics can be
           retrieved by calling `logger.get_duration_stats(logging_file_path)`. <br>
           Be aware that the `logging_file_path` gets cleared before the logger pushes
           logged messages. Hence, one should use a separate logging file for each
           constructed processing and feature instance with this library.
        * It is not possible to pass the `return_df` argument from the `process`
          method, because this should always be True (in order to output a valid
          iterable in the `transform` method)

        """
        self.processors = processors
        self.return_all_signals = return_all_signals
        self.drop_keys = drop_keys
        self.logging_file_path = logging_file_path

    def fit(self, X=None, y=None) -> SKSeriesPipeline:
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
        SKSeriesPipeline
            The transformer instance itself is returned.
        """
        return self

    def transform(
        self,
        X: Union[List[Union[pd.Series, pd.DataFrame]], pd.Series, pd.DataFrame],
    ) -> pd.DataFrame:
        """Process the signals.

        Parameters
        ----------
        X: : Union[List[Union[pd.Series, pd.DataFrame]], pd.Series, pd.DataFrame]
            The signals on which the preprocessing steps will be executed. The signals
            need a datetime index.

        Returns
        -------
        pd.DataFrame
            A DataFrame, containing the processed signals.

        """
        processing_pipeline = SeriesPipeline(self.processors)
        return processing_pipeline.process(
            signals=X,
            return_all_signals=self.return_all_signals,
            return_df=True,
            drop_keys=self.drop_keys,
            logging_file_path=self.logging_file_path,
        )

    def __repr__(self) -> str:
        """Representation string of a SKSeriesPipeline."""
        return repr(SeriesPipeline(self.processors))
