"""SeriesPipeline class for time-series data (pre-)processing pipeline."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import itertools
import dill
import logging
import warnings

from ..utils.data import series_dict_to_df, to_series_list
from .series_processor import SeriesProcessor
from .logger import logger


class _ProcessingError(Exception):
    pass


class SeriesPipeline:
    """Pipeline for applying `SeriesProcessor` objects sequentially."""

    def __init__(self, processors: Optional[List[SeriesProcessor]] = None):
        """Create a `SeriesPipeline` instance.

        Parameters
        ----------
        processors : List[SeriesProcessor], optional
            List of `SeriesProcessor` objects that will be applied sequentially to the
            internal series dict, by default None. **The processing steps will be
            executed in the same order as passed in this list**.

        """
        self.processing_steps: List[SeriesProcessor] = [] # TODO: dit private of niet?
        if processors is not None:
            self.processing_steps = processors

    def get_required_series(self) -> List[str]:
        """Return all required series names for this pipeline.

        Return the list of series names that are required in order to execute all the
        `SeriesProcessor` objects of this processing pipeline.

        Returns
        -------
        List[str]
            List of all the required series names.

        """
        flatten = itertools.chain.from_iterable
        return list(
            set(
                flatten(
                    [step.required_series for step in self.processing_steps]
                )
            )
        )

    def append(self, processor: SeriesProcessor) -> None:
        """Append a `SeriesProcessor` at the end of the pipeline.

        Parameters
        ----------
        processor : SeriesProcessor
            The `SeriesProcessor` that will be added to the end of the pipeline

        """
        self.processing_steps.append(processor)

    def insert(self, idx: int, processor: SeriesProcessor) -> None:
        """Insert a `SeriesProcessor` at the given index in the pipeline.

        Parameters
        ----------
        idx : int
            The index where the given processor should be inserted in the pipeline.
            Index 0 will insert the given processor at the front of the pipeline,
            and index `len(pipeline)` is equivalent to appending the processor.
        processor : SeriesProcessor
            The `SeriesProcessor` that will be added to the end of the pipeline

        """
        self.processing_steps.insert(idx, processor)

    def process(
        self,
        data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        return_all_series: Optional[bool] = True,
        return_df: Optional[bool] = True,
        drop_keys: Optional[List[str]] = None,
        logging_file_path: Optional[Union[str, Path]] = None,
    ) -> Union[List[pd.Series], pd.DataFrame]:
        """Execute all `SeriesProcessor` objects in pipeline sequentially.

        Apply all the processing steps on passed Series list or DataFrame and return the
        preprocessed Series list or DataFrame.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            Dataframe or Series or list thereof, with all the required data for the
            processing steps. \n
            **Remark**: each Series/DataFrame must have a `pd.DatetimeIndex`.
        return_all_series : bool, optional
            Whether the output needs to return all the series, by default True.
            If `True` the output will contain all series that were passed to this
            method. If `False` the output will contain just the required series (see
            `get_required_series`).
        return_df : bool, optional
            Whether the output needs to be a series dict or a DataFrame, default True.
            If `True` the output series will be combined to a DataFrame with an outer
            merge.
        drop_keys : List[str], optional
            Which keys should be dropped when returning the output, by default None.
        logging_file_path : Union[str, Path], optional
            The file path where the logged messages are stored, by default None.
            If `None`, then no logging `FileHandler` will be used and the logging
            messages are only pushed to stdout. Otherwise, a logging `FileHandler` will
            write the logged messages to the given file path.

        Returns
        -------
        Union[List[pd.Series], pd.DataFrame]
            The preprocessed series.

        Notes
        -----
        * If a `logging_file_path` is provided, the execution (time) info can be
          retrieved by calling `logger.get_processor_logs(logging_file_path)`. <br>
          Be aware that the `logging_file_path` gets cleared before the logger pushes
          logged messages. Hence, one should use a separate logging file for each
          constructed processing and feature instance with this library.
        * If a series processor its function output is a `np.ndarray`, the input series
          dict (required dict for that function) must contain just 1 series! That series
          its name and index are used to return a series dict. When a user does not want
          a numpy array to replace its input series, it is his / her responsibility to
          create a new `pd.Series` (or `pd.DataFrame`) of that numpy array with a
          different (column) name.
        * If `func_output` is a `pd.Series`, keep in mind that the input series gets
          transformed (i.e., replaced) in the pipeline with the `func_output` when the
          series name is  equal.

        Raises
        ------
        _ProcessingError
            Error raised when a processing step fails.

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
                    "This file will be overwritten!"
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
            assert type(s) == pd.Series, f"Error non pd.Series object passed: {type(s)}"
            if not return_all_series:
                # If just the required series have to be returned
                if s.name in self.get_required_series():
                    series_dict[str(s.name)] = s.copy()
            else:
                # If all the series have to be returned
                series_dict[str(s.name)] = s.copy()

        output_keys = set()  # Maintain set of output series
        for processor in self.processing_steps:
            try:
                processed_dict = processor(series_dict)
                output_keys.update(processed_dict.keys())
                series_dict.update(processed_dict)
            except Exception as e:
                raise _ProcessingError(
                    "Error while processing function {}:\n {}".format(
                        processor.name, str(e)
                    )
                ) from e

        if not return_all_series:
            # Return just the output series
            output_dict = {key: series_dict[str(key)] for key in output_keys}
            series_dict = output_dict

        if drop_keys is not None:
            # Drop the keys that should not be included in the output
            output_dict = {
                key: series_dict[key]
                for key in set(series_dict.keys()).difference(drop_keys)
            }
            series_dict = output_dict

        if return_df:
            # We merge the series dict into a DataFrame
            return series_dict_to_df(series_dict)
        else:
            return [s for s in series_dict.values()]

    def serialize(self, file_path: Union[str, Path]):
        """Serialize this `SeriesProcessor` instance.

        Note
        ----
        As we use `dill` to serialize, we can also serialize (decorator)functions which
        are defined in the local scope, like lambdas.

        Parameters
        ----------
        file_path : Union[str, Path]
            The path where the `SeriesProcessor` will be serialized.

        See Also
        --------
        https://github.com/uqfoundation/dill

        """
        with open(file_path, "wb") as f:
            dill.dump(self, f, recurse=True)

    def __repr__(self):
        """Return formal representation of object."""
        return (
            "[\n" + "".join([f"\t{str(p)}\n" for p in self.processing_steps]) + "]"
        )

    def __str__(self):
        """Return informal representation of object."""
        return self.__repr__()
