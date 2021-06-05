"""Code for signals preprocessing pipeline."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import dill
import logging
import warnings

from .series_processor import SeriesProcessor, _series_dict_to_df
from .logger import logger


class _ProcessingError(Exception):
    pass


class SeriesProcessorPipeline:
    """Pipeline containing `SeriesProcessor` object to be applied sequentially."""

    def __init__(self, processors: Optional[List[SeriesProcessor]] = None):
        """Init `SeriesProcessorPipeline object.

        Parameters
        ----------
        processors : List[SeriesProcessor], optional
            List of `SeriesProcessor` objects that will be applied sequentially to the
            signals dict, by default None. The processing steps will be executed in the 
            same order as passed with this list.

        """
        self.processing_registry: List[SeriesProcessor] = []
        if processors is not None:
            self.processing_registry = processors

    def get_all_required_signals(self) -> List[str]:
        """Return required signal for this pipeline.

        Return a list of signal keys that are required in order to execute all the
        `SeriesProcessor` objects that currently are in the pipeline.

        Returns
        -------
        List[str]
            List of all the required signal keys.

        """
        return list(
            set(
                chain.from_iterable(
                    [pr.required_series for pr in self.processing_registry]
                )
            )
        )

    def append(self, processor: SeriesProcessor) -> None:
        """Append a `SeriesProcessor` at the end of pipeline.

        Parameters
        ----------
        processor : SeriesProcessor
            The `SeriesProcessor` that will be added to the end of the pipeline

        """
        self.processing_registry.append(processor)

    def __call__(
        self,
        signals: Union[
            List[Union[pd.Series, pd.DataFrame]],
            pd.Series,
            pd.DataFrame,
        ],
        return_all_signals: Optional[bool] = True,
        return_df: Optional[bool] = True,
        drop_keys: Optional[List[str]] = [],
        logging_file_path: Optional[Union[str, Path]] = None,
    ) -> Union[Dict[str, pd.Series], pd.DataFrame]:
        """Execute all `SeriesProcessor` objects in pipeline sequentially.

        Apply all the processing steps on passed Series list or DataFrame and return the
        preprocessed Series list or DataFrame.

        Parameters
        ----------
        signals : Union[List[Union[pd.Series, pd.DataFrame]], pd.Series, pd.DataFrame]
            The signals on which the preprocessing steps will be executed. The signals
            need a datetime index.
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

        Returns
        -------
        Union[Dict[str, pd.Series], pd.DataFrame]
            The preprocessed series.

        Note
        ----
        If a `logging_file_path` is provided, the execution (time) statistics can be
        retrieved by calling `logger.get_function_duration_stats(logging_file_path)` and
        `logger.get_key_duration_stats(logging_file_path)`.
        Be aware that the `logging_file_path` gets cleared before the logger pushes 
        logged messages. Hence, one should use a separate logging file for the 
        processing and the feature part of this library.

        Raises
        ------
        _ProcessingError
            Error raised when a processing step fails.

        Note
        ----
        If a series processor its function output is a `np.ndarray`, the input series
        dict (required dict for that function) must contain just 1 series! That series
        its name and index are used to return a series dict. When a user does not want a
        numpy array to replace its input series, it is his / her responsibility to
        create a new `pd.Series` (or `pd.DataFrame`) of that numpy array with a
        different (column) name.
        If `func_output` is a `pd.Series`, keep in mind that the input series gets
        transformed (i.e., replaced) in the pipeline with the `func_output` when the
        series name is  equal.

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

        # Converting the signals list into a dict
        series_dict = dict()

        def to_list(x):
            if not isinstance(x, list):
                return [x]
            return x

        series_list = []
        for series in to_list(signals):
            if type(series) == pd.DataFrame:
                series_list += [series[c] for c in series.columns]
            else:
                assert isinstance(series, pd.Series)
                series_list.append(series)

        for s in series_list:
            assert type(s) == pd.Series, f"Error non pd.Series object passed: {type(s)}"
            if not return_all_signals:
                # If just the required signals have to be returned
                if s.name in self.get_all_required_signals():
                    series_dict[s.name] = s.copy()
            else:
                # If all the signals have to be returned
                series_dict[s.name] = s.copy()

        output_keys = set()  # Maintain set of output signals
        for processor in self.processing_registry:
            try:
                processed_dict = processor(series_dict)
                output_keys.update(processed_dict.keys())
                series_dict.update(processed_dict)
            except Exception as e:
                raise _ProcessingError(
                    "Error while processing function {}".format(processor.name)
                ) from e

        if not return_all_signals:
            # Return just the output signals
            output_dict = {key: series_dict[key] for key in output_keys}
            series_dict = output_dict

        if drop_keys:
            # Drop the keys that should not be included in the output
            output_dict = {
                key: series_dict[key]
                for key in set(series_dict.keys()).difference(drop_keys)
            }
            series_dict = output_dict

        if return_df:
            # We merge the signals dict into a DataFrame
            return _series_dict_to_df(series_dict)
        else:
            return series_dict

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
            "[\n" + "".join([f"\t{str(pr)}\n" for pr in self.processing_registry]) + "]"
        )

    def __str__(self):
        """Return informal representation of object."""
        return self.__repr__()
