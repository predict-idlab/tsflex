"""Code for signals preprocessing pipeline."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

from itertools import chain
from typing import Dict, List, Union

import pandas as pd


def dataframe_func(func):
    """Decorate function to use a DataFrame instead of a series dict.

    This decorator can be used for functions that need to work on a
    whole DataFrame, it will convert the series dict into a DataFrame with an outer
    merge. The decorated function has to take a DataFrame as first
    argument and also return a DataFrame.
    The function's prototype should be:
    "func(df : pd.DataFrame, **kwargs) -> pd.DataFrame"
    """

    def wrapper(series_dict: Dict[str, pd.Series], **kwargs):
        df = _series_dict_to_df(series_dict)
        res = func(df, **kwargs)
        return res

    return wrapper


def single_series_func(func):
    """Decorate function to use single Series instead of a series dict.

    The signals dict will be passed as multiple signals to the decorated function.
    This should be in function where the key has no importance and the processing
    can be applied to all the required signals identically. The decorated function
    has to take a pandas Series as input and also return a pandas Series.
    The function's prototype should be:
    "func(signals: pd.Series, **kwargs) -> pd.Series"
    """

    def wrapper(series_dict: Dict[str, pd.Series], **kwargs):
        output_dict = dict()
        for k, v in series_dict.items():
            output_dict[k] = func(v, **kwargs)

        return output_dict

    return wrapper


def _series_dict_to_df(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Convert the `series_dict` into a pandas DataFrame with an outer merge.

    Parameters
    ----------
    series_dict : Dict[str, pd.Series]
        The dict with the Series.

    Returns
    -------
    pd.DataFrame
        The merged pandas DataFrame
    """
    df = pd.DataFrame()
    for _, s in series_dict.items():
        df = df.merge(s, left_index=True, right_index=True, how="outer")
    return df


class _ProcessingError(Exception):
    pass


class SeriesProcessor:
    """Class that executes a specific operation on the passed series_dict."""

    def __init__(self, required_series: List[str], func, name=None, **kwargs):
        """Init a SeriesProcessor object.

        Parameters
        ----------
        required_series : List[str]
            A list of the required signals for this processor.
        func : Callable
            A callable that does the processing on a series_dict. `func` has
            to take a dict with keys the signal names and the corresponding
            (time indexed) Series as input. It has to output the processed
            series_dict. The prototype of the function should match:
            `func(series_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]`.
        name : str, optional
            The name of the processor, by default None and the `func.__name__`
            will be used.
        """
        self.required_series = required_series
        self.func = func
        if name:
            self.name = name
        else:
            self.name = self.func.__name__

        self.kwargs = kwargs

    def __call__(self, series_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Cal(l)culates the processed signal.

        Parameters
        ----------
        series_dict : Dict[str, pd.Series]
            A dict of pandas signals containing the signals that neet to be processed.

        Returns
        -------
        Dict[str, pd.Series]
            The processed `series_dict`

        Raises
        ------
        KeyError
            Raised when a key is not present in the `series_dict` but required for the
            processing.
        """
        # Only selecting the signals that are needed for this processing step
        requested_dict = {}
        try:
            for sig in self.required_series:
                requested_dict[sig] = series_dict[sig]
        except KeyError as key:
            # Re raise error as we can't continue
            raise KeyError(
                "Key %s is not present in the input dict and needed for processor %s"
                % (key, self.name)
            )

        return (
            self.func(requested_dict, **self.kwargs)
            if self.kwargs is not None
            else self.func(series_dict)
        )

    def __repr__(self):
        """Return formal representation of object."""
        return self.name + (" " + str(self.kwargs)) if self.kwargs is not None else ""

    def __str__(self):
        """Return informal representation of object."""
        return self.__repr__()


class SeriesProcessorPipeline:
    """Pipeline containing `SeriesProcessor` object to be applied sequentially."""

    def __init__(self, processors: List[SeriesProcessor] = []):
        """Init `SeriesProcessorPipeline object.

        Parameters
        ----------
        processors : List[SeriesProcessor], optional
            List of `SeriesProcessor` objects that will be applied sequentially to the
            signals dict. The processing steps will be executed in the same order as
            passed with this list, by default []
        """
        self.processing_registry: List[SeriesProcessor] = processors

    def get_all_required_signals(self) -> List[str]:
        """Return required signal for this pipeline.

        Return a list of signal keys that are required in order to exectue all the
        `SeriesProcessor` objects that currently are in the pipeline.
        Returns
        -------
        List[str]
            List of all the required singal keys.
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
        self, signals: Union[List[pd.Series], pd.DataFrame], return_df=True
    ) -> Union[Dict[str, pd.Series], pd.DataFrame]:
        """Execute all `SeriesProcessor` obects in pipeline sequentially.

        Apply all the processing steps on passed Series list or DataFrame and return the
        preprocessed Series list or DataFrame.
        Parameters
        ----------
        signals : Union[List[pd.Series], pd.DataFrame]
            The signals on which the preprocessing steps will be executed. The signals
            need a datetime index.
        return_df : bool, default: True
            Wether the output needs to be a series dict or a DataFrame. If `True` the
            output series will be combined to a DataFrame with an outer merge,
            by default True.

        Returns
        -------
        Union[Dict[str, pd.Series], pd.DataFrame]
            The preprocessed series.

        Raises
        ------
        _ProcessingError
            Error raised when a processing step fails.
        """
        # Converting the signals list into a dict
        series_dict = dict()

        if type(signals) == pd.DataFrame:
            series_list = [signals[s] for s in self.get_all_required_signals()]
        else:
            series_list = signals

        for s in series_list:
            assert type(s) == pd.Series, "Error non pd.Series object passed"
            series_dict[s.name] = s.copy()

        for processor in self.processing_registry:
            try:
                series_dict.update(processor(series_dict))
            except Exception as e:
                raise _ProcessingError(
                    "Error while processing function {}".format(processor.name)
                ) from e

        if return_df:
            # We merge the signals dict into a DataFrame
            return _series_dict_to_df(series_dict)
        else:
            return series_dict

    def __repr__(self):
        """Return formal representation of object."""
        return (
            "[\n" + "".join([f"\t{str(pr)}\n" for pr in self.processing_registry]) + "]"
        )

    def __str__(self):
        """Return informal representation of object."""
        return self.__repr__()
