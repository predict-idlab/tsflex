"""Code for signals preprocessing."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import time
import warnings

from ..utils.series_dict import series_dict_to_df
from .logger import logger


def dataframe_func(func: Callable):
    """Decorate function to use a DataFrame instead of a series dict. # TODO dit beter uitleggen (eg waarom series_dict vaal voldoende is)

    This decorator can be used for functions that need to work on a whole DataFrame,
    it will convert the series dict input into a DataFrame with an outer merge.
    The decorated function has to take a DataFrame as first argument.
    The function's prototype should be:
    "func(df : pd.DataFrame, **kwargs)
        -> Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]"
    """

    def wrapper(series_dict: Dict[str, pd.Series], **kwargs):
        df = series_dict_to_df(series_dict)
        res = func(df, **kwargs)
        return res

    wrapper.__name__ = "dataframe_func: " + func.__name__
    return wrapper


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
    The given `np_array` receives the same index and name as the `series`.
    Hence, the `np_array` needs to have the same length as the `series`.
    Giving the `np_array` the same name as the `series`, will result in transforming
    (i.e., replacing) the `series` in the pipeline.
    When a user does not want a numpy array to replace its input series, it is his / her
    responsibility to create a new `pd.Series` (or `pd.DataFrame`) of that numpy array
    with a different (column) name.

    """
    # The length of the out has to be the same as the signal length
    assert len(np_array) == len(series)
    return pd.Series(data=np_array, index=series.index, name=series.name)


def _handle_seriesprocessor_func_output(
    func_output: Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]],
    required_dict: Dict[str, pd.Series],
    func_name: str,
) -> Union[Dict[str, pd.Series], pd.DataFrame]:
    """Handle the output of a SeriesProcessor its function.

    Parameters
    ----------
    func_output: Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]
        The output of the SeriesProcessor its function.
    required_dict: Dict[str, pd.Series]
        The series dict that contains the required signals for the `SeriesProcessor` its
        function.
    func_name: str
        The name of the SeriesProcessor (its function).

    Returns
    -------
    Union[Dict[str, pd.Series], pd.DataFrame]
        The processed outputs in the given func_output.
        If the `func_output` is a `pd.DataFrame`, the `pd.DataFrame` is returned.
        If the `func_output` is a `pd.Series`, `np.ndarray`, or list of `pd.Series`,
        a series dict is returned.

    Raises
    ------
    TypeError
        Error raised when function output is invalid.

    Note
    ----
    If `func_output` is a `np.ndarray`, the given `requested_dict` must contain just 1
    series! That series its name and index are used to return a series dict. When a
    user does not want a numpy array to replace its input series, it is his / her
    responsibility to create a new `pd.Series` (or `pd.DataFrame`) of that numpy array
    with a different (column) name.
    If `func_output` is a `pd.Series`, keep in mind that the input series gets
    transformed (i.e., replaced) with the `func_output` when the series name is equal.

    """
    if isinstance(func_output, pd.DataFrame):
        # Nothing has to be done! A pd.DataFrame can be added to a series_dict using
        # series_dict.update(df)
        # Note: converting this to a dictionary (to_dict()) is **very** inefficient!
        return func_output
    elif isinstance(func_output, pd.Series):
        # Convert series to series_dict and return
        if len(required_dict) == 1:
            # In a series_dict input_key == series.name
            input_key = list(required_dict.keys())[0]
            if input_key != func_output.name:
                # TODO: unsure about this warning
                warnings.warn(
                    "Function output is a single series with a different name "
                    + f"({func_output.name}) from the input series name {input_key}!\n"
                    + "\t > Make sure this is expected behavior! Input signal "
                    + f"{input_key} won't be updated with the function output, instead "
                    + f"output {func_output.name} will be appended to the outputs."
                )
        return {func_output.name: func_output}
    elif isinstance(func_output, np.ndarray):
        # Must be constructed from just 1 signal
        assert len(required_dict) == 1
        input_signal = list(required_dict.values())[0]
        return {input_signal.name: _np_array_to_series(func_output, input_signal)}
    elif isinstance(func_output, list):
        # Nothing has to be done! A dict can be directly added to the series_dict
        assert len(set([s.name for s in func_output])) == len(func_output)
        return {s.name: s for s in func_output}
    else:
        raise TypeError(f"Function output type is invalid for processor {func_name}")


def _handle_single_series_func(
    func: Callable,
    required_dict: Dict[str, pd.Series],
    func_name: str,
    **kwargs,
) -> Dict[str, pd.Series]:
    """Handle a function that uses a single series instead of a series dict.

    This is a wrapper for a function that requires a `pd.Series` as input.
    The signals of the `required_dict` are passed one-by-one to the function and the
    output is aggregated in a (new) series_dict.

    Parameters
    ----------
    func: Callable[[pd.Series], Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]]
        The output of the SeriesProcessor its function.
    func : Callable
        A callable that processes a single series. `func` has to take a Series as input.
        The output can be rather versatile.
        The prototype of the function should match:
        `func(series: pd.Series)
            -> Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]`.
    required_dict: Dict[str, pd.Series]
        The series dict that contains the required signals for the SeriesProcessor its
        function.
    func_name: str
        The name of the SeriesProcessor (its function).

    Returns
    -------
    Dict[str, pd.Series]
        The processed outputs for the given `required_dict`.

    Raises
    ------
    TypeError
        Error raised when a processing step fails.

    Note
    ----
    If you want to transform (i.e., replace) the input series in the pipeline, than
    `func` should return either:
        * a `np.ndarray`.
        * a `pd.Series` with the same name as the input series.
        * a `pd.DataFrame` with (one) column name equal to the input series its name.
        * a list of `pd.Series` in which (exact) one series has the same name as the
          input series.
    Series (& columns) with other (column) names will be added to the series dict.

    """
    output_dict = dict()
    for k, v in required_dict.items():
        func_output = func(v, **kwargs)
        # Handle the function output (i.e., convert to correct type)
        func_output = _handle_seriesprocessor_func_output(
            func_output, {k: v}, func_name
        )
        # Check that the output of the function call produces unique columns / keys
        assert len(set(output_dict.keys()).intersection(func_output.keys())) == 0
        output_dict.update(func_output)
    return output_dict

class SeriesProcessor:
    """Class that executes a specific operation on the passed series_dict."""

    def __init__(
        self,
        required_series: List[str],
        func: Callable,
        single_series_func: Optional[bool] = False,
        name: Optional[str] =None,
        **kwargs,
    ):
        """Init a SeriesProcessor object.

        Parameters
        ----------
        required_series : List[str]
            A list of the required signals for this processor.
        func : Callable
            A callable that processes a series_dict (or a single series; see below).
            `func` has to take a dict with keys the signal names and the corresponding
            (time indexed) Series as input.
            `func` could also take a `pd.Series` as input, in that case the flag
            `single_series_func` should be set to True.
            The output can be rather versatile.
            The prototype of the function should match:
            `func(series_dict: Union[Dict[str, pd.Series], pd.Series])
                -> Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]`.
        single_series_func : bool, optional
            Whether the given `func` is a single series function, by default False.
            A single series function is a function that takes 1 series as input and
            thus has to be called for each of the `required_series`.
        name : str, optional
            The name of the processor, by default None and the `func.__name__`
            will be used.

        Note
        ----
        If the output of `func` is a `np.ndarray`, the given `required_series` must have
        length 1, i.e., the function requires just 1 series! That series its name and
        index are used to return a series dict.

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
        This internal representation is constructed in the `process` method of the
        `SeriesPipeline`.

        Note
        ----
        If you want to test or debug your `SeriesProcessor` object, just encapsulate
        your instance of this class in a `SeriesPipeline`. The latter allows more 
        versatile input for its `process` method.

        """
        t_start = time.time()

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

        # Variable that will contain the final output of this method
        processed_output: Union[Dict[str, pd.Series], pd.DataFrame]
        if self.single_series_func:
            # Handle series func includes _handle_seriesprocessor_func_output!
            processed_output =  _handle_single_series_func(
                self.func, requested_dict, self.name, **self.kwargs
            )
        else:
            func_output = self.func(requested_dict, **self.kwargs)
            processed_output = _handle_seriesprocessor_func_output(func_output, requested_dict, self.name)

        elapsed = time.time() - t_start
        logger.info(
                f"Finished function [{self.name}] as [single_series_func={self.single_series_func}] on {list(requested_dict.keys())} in [{elapsed} seconds]!"
        )

        return processed_output

    def __repr__(self):
        """Return formal representation of object."""
        return self.name + (" " + str(self.kwargs)) if self.kwargs is not None else ""

    def __str__(self):
        """Return informal representation of object."""
        return self.__repr__()
