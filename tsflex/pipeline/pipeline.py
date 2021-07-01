"""Pipeline class that wraps sklearn.pipeline."""

__author__ = "Jeroen Van Der Donckt"

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from typing import List, Tuple, Union, Optional

from ..processing import SKSeriesPipeline
from ..features import SKFeatureCollection
from .dataframe_operator import to_dataframe_operator


def make_pipeline(
    steps: List[Tuple[str, Union[TransformerMixin, BaseEstimator]]],
    memory: Optional[Union[str, joblib.Memory]] = None,
    verbose: Optional[bool] = False,
):
    """Make a sklearn-`Pipeline` of given (named) steps containing tsflex components.

    This method creates a pipeline that may contain `SKSeriesPipeline` and 
    `SKFeatureCollection` steps. Concretely, the pipeline wraps all operators in a 
    `DataFrameOperator`, in order to retain the data its column names.

    Note
    ----
    When calling the methods of the returned pipeline, the `X` argument should always
    be a pandas DataFrame.

    Parameters
    ----------
    steps : List[Tuple[str, Union[TransformerMixin, BaseEstimator]]]
        List of (name, transform) tuples (implementing fit/transform) that are chained, 
        in the order in which they are chained, with the last object an estimator.
    memory : Union[str, joblib.Memory], optional
        Used to cache the fitted transformers of the pipeline, by default None. 
        If None, no caching is performed. If a string is given, it is the path to the 
        caching directory. Enabling caching triggers a clone of the transformers before 
        fitting. Therefore, the transformer instance given to the pipeline cannot be 
        inspected directly. Use the attribute ``named_steps`` or ``steps`` to inspect 
        estimators within the pipeline. Caching the transformers is advantageous when 
        fitting is time consuming.
    verbose : bool, optional
        Whether the time elapsed while fitting each step should be printed, by default
        False.

    """
    # Wrap the steps of the pipeline into a DataFrameOperator if necessary
    def wrap_step(step):
        # It is not necessary to wrap SKSeriesPipeline or SKFeatureCollection as
        # these transformers return a dataframe by default :)
        if (
            isinstance(step[1], SKSeriesPipeline) or 
            isinstance(step[1], SKFeatureCollection)
        ):
            return step
        return (step[0], to_dataframe_operator(step[1]))

    steps = [wrap_step(step) for step in steps]
    return Pipeline(steps, memory=memory, verbose=verbose)
