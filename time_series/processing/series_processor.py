"""Code for signals preprocessing pipeline."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

from itertools import chain
from typing import Callable, Dict, List, Union, Optional

import pandas as pd
import numpy as np


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
        assert isinstance(res, pd.DataFrame)
        return res

    wrapper.__name__ = "[wrapped: dataframe_func] " + func.__name__
    return wrapper


# def numpy_func(func):
#     """Decorate function to use numpy array instead of a series dict.

#     The signals dict will be passed as multiple signals to the decorated function.
#     This should be a function where the key has no importance and the processing
#     can be applied to all the required signals identically. The decorated function
#     has to take a numpy array as input and also return a numpy array.
#     The function's prototype should be:
#     "func(signal: np.ndarray, **kwargs) -> np.ndarray"
#     """

#     def wrapper(series_dict: Dict[str, pd.Series], **kwargs):
#         output_dict = dict()
#         for k, v in series_dict.items():
#             res = func(v.values, **kwargs)
#             assert isinstance(res, np.ndarray)
#             res = pd.Series(data=res, index=v.index, name=v.name)
#             output_dict[k] = res

#         return output_dict

#     wrapper.__name__ = "[wrapped: numpy_func] " + func.__name__
#     return wrapper


def _handle_single_series_func(
    func: Callable[[pd.Series], Union[pd.Series, np.ndarray]],
    required_dict: Dict[str, pd.Series],
    **kwargs,
) -> Union[Dict[str, pd.Series], pd.DataFrame]:
    """Handle a function that uses a single Series instead of a series dict.

    This is a wrapper for a function that requires a `pd.Series` as input and processes
    that input signal.
    The signals of the  `required_dict` are passed one-by-one to the function and the 
    output is aggregated in a (new) series_dict.

    There are two possible cases (for the function);
    1. The single output case
       => The `func` transforms the input signal to a `pd.Series` or `np.ndarray`. 
          The output is in this case aggregated at the key of the input signal.
          In this case the function's prototype should be:
            `func(series_dict: Dict[str, pd.Series]) 
                -> Union[np.ndarray, pd.Series]`.
    2. The multi output case
       => The `func` produces a multi-output format, i.e., a `pd.DataFrame` or 
          `series_dict`.
          The output is in this  case aggregated at the columns / keys of the 
          produced multi-output format.
          In this case the function's prototype should be:
            `func(series_dict: Dict[str, pd.Series]) 
                -> Union[pd.DataFrame, Dict[str, pd.Series]]`.


    Note
    ----
    In the single output case (1.), the function `func` transforms a signal, hence the 
    function should output a single series or array (this array should have the same 
    length as the input signal).
    In the multi output case (2.), the function `func` produces a new output based on
    the passed signal, hence the  function should output a dataframe or series_dict.
    !! Note that when the func is called multiple times (when len(required_dict) > 1),
       than should the multi-output have distinct keys for each function call.
       => Thus it is the end-user its responsability to have `func` output different 
          columns / keys depending on the input signal (when len(required_dict) > 1).
       # TODO: add this note also at the Series Processor?

    """
    output_dict = dict()
    nb_required_signals = len(required_dict)
    for k, v in required_dict.items():
        func_output = func(v, **kwargs)
        # 1. Single output case
        if isinstance(func_output, pd.Series):
            output_dict[k] = func_output
        elif isinstance(func_output, np.ndarray):
            output_dict[k] = _np_array_to_series(func_output, v)
        # 2. Multi output case
        elif isinstance(func_output, pd.DataFrame) or isinstance(func_output, dict):
            # 2.1 If there is just 1 signal => return the dataframe / dict
            if nb_required_signals == 1:
                return func_output
            # 2.2 Else update the output dict
            # Check that the output of the function call produces unique columns / keys
            assert len(set(output_dict.keys()).intersection(func_output.keys())) == 0
            output_dict.update(func_output)
        else:
            raise TypeError(
                f"Function output type is invalid for function {func.__name__}"
            )
    return output_dict


def _df_dict_to_series_list(
    df_dict: Dict[str, Union[pd.DataFrame, pd.Series]]
) -> List[pd.Series]:
    """Convert a DataFrame dict to a list of Series.

    Parameters
    ----------
    df_dict : Dict[str, Union[pd.DataFrame, pd.Series]]
        The series dict that will be converted.

    Returns
    -------
    List[pd.Series]
        A list of series.

    """
    series_list = []
    for dfs in df_dict.values():
        if isinstance(dfs, pd.Series):
            series_list.append(dfs)
        elif isinstance(dfs, pd.DataFrame):
            series_list += [dfs[c] for c in dfs.columns]
    return series_list


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

    Note
    ----
    The `series_dict` is an internal representation of the signals list.
    In this dictionary, the key is always the accompanying series its name.
    This internal representation is constructed in the `__call__` method of the
    `SeriesProcessorPipeline`.

    """
    # 0. Check if the series_dict has only 1 signal, to create the df efficiently
    if len(series_dict) == 1:
        return pd.DataFrame(series_dict)
    # 1. Check if the time-indexes of the series are equal, to create the df efficiently
    index_info = set(
        [(s.index[0], s.index[-1], len(s), s.index.freq) for s in series_dict.values()]
    )
    freq_idx = -1  # The index of the frequency info in the tuple(s) in index_info
    if len(index_info) == 1 and list(index_info)[0][freq_idx] is not None:
        # When the time-indexes are the same we can create df very efficiently
        return pd.DataFrame(series_dict)
    # 2. If check failed, create the df by merging the series (the slow way)
    df = pd.DataFrame()
    for s in series_dict.values():
        df = df.merge(s, left_index=True, right_index=True, how="outer")
    return df


def _np_array_to_series(np_array: np.ndarray, series: pd.Series) -> pd.Series:
    """Convert the `np_array` into a pandas Series.

    Parameters
    ----------
    np_array: np.ndarray
        The numpy array that needs to be converted to a pandas Series.
    series: pd.Series
        The pandas Series that contains the name and the index for the `np_array`.

    Returns
    -------
    pd.Series
        The numpy array as a pandas Series with as index the given series its index and
        as name the series its name

    Note
    ----
    This method requires the `np_array` to have the same length as the `series`.

    """
    # The length of the out has to be the same as the signal length
    assert len(np_array) == len(series)
    # TODO: check if func_output ndim is correct
    return pd.Series(data=np_array, index=series.index, name=series.name)


class _ProcessingError(Exception):
    pass


class SeriesProcessor:
    """Class that executes a specific operation on the passed series_dict."""

    def __init__(
        self,
        required_series: List[str],
        func,
        single_series_func=False,
        name=None,
        **kwargs,
    ):
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
            `func(series_dict: Dict[str, pd.Series])
                -> Union[np.ndarray, pd.Series, pd.DataFrame, Dict[str, pd.Series]]`.
        single_series_func : bool, optional
            Whether the given `func` is a single series function, by default False.
            A single series function is a function that takes 1 series as input and
            thus has to be called for each of the `required_series`.
        name : str, optional
            The name of the processor, by default None and the `func.__name__`
            will be used.

        """
        self.required_series = required_series
        self.func = func
        self.single_series_func = single_series_func
        if name:
            self.name = name
        else:
            self.name = self.func.__name__

        self.kwargs = kwargs

    def __call__(
        self, series_dict: Dict[str, pd.Series]
    ) -> Union[Dict[str, pd.Series], pd.DataFrame]:
        """Cal(l)culates the processed signal.

        Parameters
        ----------
        series_dict : Dict[str, pd.Series]
            A dict of pandas signals containing the signals that need to be processed.
            The key should always be the accompanying series its name.

        Returns
        -------
        Dict[str, pd.Series]
            The processed `series_dict`

        Raises
        ------
        KeyError
            Raised when a key is not present in the `series_dict` but required for the
            processing.
        TypeError
            Raised when the output of the `SeriesProcessor` is not of the correct type.

        Note
        ----
        The `series_dict` is actually an internal representation of the signals list.
        This internal representation is constructed in the `__call__` method of the
        `SeriesProcessorPipeline`.

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

        def call_func():
            if self.single_series_func:
                return  _handle_single_series_func(self.func, requested_dict, **self.kwargs)
            return self.func(requested_dict, **self.kwargs)

        func_output = call_func()

        if isinstance(func_output, pd.DataFrame):
            # Nothing has to be done! A pd.DataFrame can be added to a series_dict using
            # series_dict.update(df)
            # Note: converting this to a dictionary (to_dict()) is **very** inefficient!
            return func_output
        elif isinstance(func_output, pd.Series):
            # Convert series to series_dict and return
            return {func_output.name, func_output}
        elif isinstance(func_output, np.ndarray):
            # Must be constructed from just 1 signal
            assert len(requested_dict) == 1
            input_signal = list(requested_dict.values())[0]
            return {input_signal.name: _np_array_to_series(func_output, input_signal)}
        elif isinstance(func_output, dict):
            # Nothing has to be done! A dict can be directly added to the series_dict
            return func_output
        else:
            raise TypeError(
                f"Function output type is invalid for processor {self.name}"
            )

    def __repr__(self):
        """Return formal representation of object."""
        return self.name + (" " + str(self.kwargs)) if self.kwargs is not None else ""

    def __str__(self):
        """Return informal representation of object."""
        return self.__repr__()


class SeriesProcessorPipeline:
    """Pipeline containing `SeriesProcessor` object to be applied sequentially."""

    def __init__(self, processors: Optional[List[SeriesProcessor]] = None):
        """Init `SeriesProcessorPipeline object.

        Parameters
        ----------
        processors : List[SeriesProcessor], optional
            List of `SeriesProcessor` objects that will be applied sequentially to the
            signals dict. The processing steps will be executed in the same order as
            passed with this list, by default None

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
            Dict[str, Union[pd.Series, pd.DataFrame]],
            List[Union[pd.Series, pd.DataFrame]],
            pd.Series,
            pd.DataFrame,
        ],
        return_all_signals=True,
        return_df=True, # TODO: drop_cols parameter??
    ) -> Union[Dict[str, Union[pd.Series, pd.DataFrame]], pd.DataFrame]:
        """Execute all `SeriesProcessor` objects in pipeline sequentially.

        Apply all the processing steps on passed Series list or DataFrame and return the
        preprocessed Series list or DataFrame.

        Parameters
        ----------
        signals : Union[Dict[str, Union[pd.Series, pd.DataFrame]], List[Union[pd.Series, pd.DataFrame]], pd.Series, pd.DataFrame]
            The signals on which the preprocessing steps will be executed. The signals
            need a datetime index.
        return_all_signals : bool, default: True
            Whether the output needs to return all the signals. If `True` the output
            will contain all signals that were passed to this method. If `False` the
            output will contain just the required signals (see
            `get_all_required_signals`).
        return_df : bool, default: True
            Whether the output needs to be a series dict or a DataFrame. If `True` the
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

        def to_list(x):
            if not isinstance(x, list):
                return [x]
            return x

        if isinstance(signals, dict):
            signals = _df_dict_to_series_list(signals)

        series_list = []
        for series in to_list(signals):
            if type(series) == pd.DataFrame:
                series_list += [series[c] for c in series.columns]
            else:
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
            # Return jus the output signals
            output_dict = {key: series_dict[key] for key in output_keys}
            series_dict = output_dict

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
