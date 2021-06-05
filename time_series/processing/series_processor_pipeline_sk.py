"""SKSeriesProcessingPipeline class for wrapped sklearn-compatible `SeriesProcessingPipeline`.

See Also
--------
Example notebooks.

"""

from __future__ import annotations   # Make typing work for the enclosing class

__author__ = "Jeroen Van Der Donckt"

import pandas as pd

from typing import List, Optional, Union
from pathlib import Path
from sklearn.base import TransformerMixin

from .series_processor import SeriesProcessor
from .processor_pipeline import ProcessorPipeline 


## Future work
# * BaseEstimator support: it is not really useful right now to support this.
#   As for example sklearn GridSearchCV requires X and y to have the same length,
#   but SeriesPipeline (sometimes) transforms the length of X in your pipeline.
#   => Possible solution; look into sklearn-contrib imblearn how they handle this


class SKSeriesProcessingPipeline(TransformerMixin):
    def __init__(
        self,
        processors: List[SeriesProcessor],
        logging_file_path: Optional[Union[str, Path]],
    ):
        """Create a `SKSeriesProcessingPipeline`.

        Parameters
        ----------

        """
        self.processors = processors
        self.logging_file_path = logging_file_path

    def fit(self, X=None, y=None) -> SKSeriesProcessingPipeline:
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
        SKSeriesProcessingPipeline
            The transformer instance itself is returned.
        """
        return self

    def transform(
        self,
        X: Union[List[Union[pd.Series, pd.DataFrame]], pd.Series, pd.DataFrame],
    ):
        processing_pipeline = ProcessorPipeline(self.processors)
        return processing_pipeline(
            signals=X, return_df=True, logging_file_path=self.logging_file_path
        )
