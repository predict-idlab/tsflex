"""Pipeline class that wraps sklearn.pipeline."""

__author__ = "Jeroen Van Der Donckt"

from sklearn.pipeline import Pipeline

from ..processing import SKSeriesPipeline
from ..features import SKFeatureCollection
from .dataframe_oprator import to_dataframe_operator


class Pipeline(Pipeline):

    def __init__(self, steps, *, memory=None, verbose=False):
        """Create a pipeline."""
        super().__init__(steps, memory=memory, verbose=verbose)
        
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

        self.steps = [wrap_step(step) for step in self.steps]
