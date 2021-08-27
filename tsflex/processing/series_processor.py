"""Code for time-series data (pre-)processing."""

__author__ = "Jonas Van Der Donckt, Emiel Deprost, Jeroen Van Der Donckt"

import time
from typing import Callable, Dict, List, Union, Tuple

import numpy as np
import pandas as pd

from .. import __pdoc__
from ..utils.classes import FrozenClass
from .logger import logger
from ..utils.data import series_dict_to_df, to_list, to_tuple, flatten

__pdoc__['SeriesProcessor.__call__'] = True


def dataframe_func(func: Callable):
    """Decorate function to use a DataFrame instead of multiple series (as argument).

    This decorator can be used for functions that need to work on a whole
    `pd.DataFrame`. It will convert the required series into a DataFrame using an
    **outer merge**.

    The function's prototype should be:

        func(df : pd.DataFrame, **kwargs)
            -> Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]

    So the decorated `func` has to take a DataFrame as first argument.

    Notes
    -----
    Only when you want to perform row-based operations, such as `df.dropna(axis=0)`,
    this wrapper is needed.
    Hence, in most cases that `func` requires a `pd.DataFrame`, series arguments would
    be sufficient; as you can perform column-based operations on multiple `pd.Series`
    (e.g., subtract 2 series) and most dataframe operations are also available for a
    `pd.Series`.

    """

    def wrapper(*series: pd.Series, **kwargs):
        series_dict = {s.name: s for s in series}
        df = series_dict_to_df(series_dict)
        res = func(df, **kwargs)
        return res

    wrapper.__name__ = "dataframe_func: " + func.__name__
    return wrapper


class SeriesProcessor(FrozenClass):
    """Class that executes a specific operation on the passed series_dict.

    Parameters
    ----------
    function : Callable
        The function that processes the series (given in the `series_names`).
        The prototype of the function should match: \n

            function(*series: pd.Series, **kwargs)
                -> Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]

        .. note::
            A function that processes a ``np.ndarray`` instead of ``pd.Series``
            should work just fine.

    series_names : Union[str, Tuple[str, ...], List[str], List[Tuple[str, ...]]]
        The names of the series on which the processing function should be applied.

        This argument should match the `function` its input; \n
        * If `series_names` is a (list of) string (or tuple of a single string),
          than `function` should require just one series as input.
        * If `series_names` is a (list of) tuple of strings, than `function` should
          require `len(tuple)` series as input **and in exactly the same order**.

        A list means multiple series (combinations) to process; \n
        * If `series_names` is a string or a tuple of strings, than `function` will
          be called only once for the series of this argument.
        * If `series_names` is a list of either strings or tuple of strings, than
          `function` will be called for each entry of this list.

        .. note::
            when passing a list as `series_names`, all items in this list should
            have the same type, i.e, either \n
            * all a `str`
            * or, all a `tuple` _with same length_. \n

    **kwargs: dict, optional
        Keyword arguments which will be also passed to the `function`

    Notes
    -----
    If the output of `function` is a `np.ndarray` or a `pd.Series` without a name,
    than (items of) the given `series_names` must have length 1, i.e., the function
    requires just 1 series! That series its name and index are used to transform
    (i.e., **replace**) that series.

    If you want to transform (i.e., **replace**) the input series with the
    processor, than `function` should return either: \n
    * a `np.ndarray` (see above).
    * a `pd.Series` with no name or with the same name as the input series.
    * a `pd.DataFrame` with (one) column name equal to the input series its name.
    * a list of `pd.Series` in which (exact) one series has the same name as the
      input series.

    Series (& columns) with other (column) names will be added to the series dict.

    """
    def __init__(
        self,
        function: Callable,
        series_names: Union[str, Tuple[str, ...], List[str], List[Tuple[str, ...]]],
        **kwargs,
    ):
        series_names = [to_tuple(names) for names in to_list(series_names)]
        # Assert that function inputs (series) all have the same length
        assert all(
            len(series_names[0]) == len(series_name_tuple)
            for series_name_tuple in series_names
        )
        self.series_names: List[Tuple[str, ...]] = series_names
        self.function = function
        self.name = self.function.__name__

        self.kwargs = kwargs
        self._freeze()

    def get_required_series(self) -> List[str]:
        """Return all required series names for this processor.

        Return the list of series names that are required in order to execute the
        processing function.

        Returns
        -------
        List[str]
            List of all the required series names.

        """
        return list(set(flatten(name for name in self.series_names)))

    def __call__(self, series_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """**Call**culates the processed series.

        Parameters
        ----------
        series_dict : Dict[str, pd.Series]
            A dict of `pd.Series` containing the data that need to be processed.
            The key should always be the accompanying series its name.

        Returns
        -------
        Dict[str, pd.Series]
            The processed `series_dict`.

        Raises
        ------
        KeyError
            Raised when a key is not present in the `series_dict` but required for the
            processing.
        TypeError
            Raised when the output of the `SeriesProcessor` is not of the correct type.

        Notes
        -----
        * The `series_dict` is an internal representation of the time-series data .
          This internal representation is constructed in the `process` method of the
          `SeriesPipeline`.
        * If you want to test or debug your `SeriesProcessor` object, just encapsulate
          your instance of this class in a `SeriesPipeline`. The latter allows more
          versatile input for its `process` method.

        """
        t_start = time.time()

        # Only selecting the series that are needed for this processing step
        # requested_dict = {}
        # try:
        #     for sig in self.get_required_series():
        #         requested_dict[sig] = series_dict[sig]
        # except KeyError as key:
        #     # Re raise error as we can't continue
        #     raise KeyError(
        #         "Key %s is not present in the input dict and needed for processor %s"
        #         % (key, self.name)
        #     )

        # Variable that will contain the final output of this method
        processed_output: Dict[str, pd.Series] = {}

        def get_series_list(keys: Tuple[str, ...]):
            """Get an ordered series list view for the given keys."""
            return [series_dict[key] for key in keys]

        def get_series_dict(keys: Tuple[str, ...]):
            """Get a series dict view for the given keys."""
            return {key: series_dict[key] for key in keys}

        for series_name_tuple in self.series_names:
            func_output = self.function(
                *get_series_list(series_name_tuple), **self.kwargs
            )
            func_output = _handle_seriesprocessor_func_output(
                func_output,
                get_series_dict(series_name_tuple),
                self.name,
            )
            # Check that the output of the function call produces unique columns / keys
            assert (
                len(set(processed_output.keys()).intersection(func_output.keys())) == 0
            )
            processed_output.update(func_output)

        elapsed = time.time() - t_start
        logger.info(
            f"Finished function [{self.name}] on {self.series_names} in "
            f"[{elapsed} seconds]!"
        )

        return processed_output

    def __repr__(self):
        """Return formal representation of object."""
        repr_str = self.name + (" " + str(self.kwargs))
        repr_str += " :  " + " ".join([str(s) for s in self.series_names])
        return repr_str

    def __str__(self):
        """Return informal representation of object."""
        return self.__repr__()


# ---------------------- utility functions for a SeriesProcessor ----------------------

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
        A view of the numpy array as a pandas Series with index the given series
        its index and name the series its name.

    Notes
    -----
    * The constructed series is a pass-by-reference `view` of the numpy array.
    * The given `np_array` receives the same index and name as the `series`.
      Hence, the `np_array` needs to have the same length as the `series`.
    * Giving the `np_array` the **same name as** the `series`, will **result in**
      transforming (i.e., **replacing**) the `series` in the pipeline.
    * When a user does not want a numpy array to replace its input series, it is
      his / her responsibility to create a `pd.Series` (or `pd.DataFrame`) of that
      numpy array with a different (column) name.

    """
    # The length of the out has to be the same as the series length
    assert len(np_array) == len(series)
    # Note: pd.Series output holds a view of np_array!
    return pd.Series(data=np_array, index=series.index, name=series.name)


def _handle_seriesprocessor_func_output(
    func_output: Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]],
    required_dict: Dict[str, pd.Series],
    func_name: str,
) -> Union[Dict[str, pd.Series], pd.DataFrame]:
    """Handle the output of a SeriesProcessor its function.

    Parameters
    ----------
    func_output : Union[np.ndarray, pd.Series, pd.DataFrame, List[pd.Series]]
        The output of the SeriesProcessor its function.
    required_dict : Dict[str, pd.Series]
        The series dict that contains the required series for the `SeriesProcessor` its
        function.
    func_name : str
        The name of the SeriesProcessor (its function).

    Returns
    -------
    Union[Dict[str, pd.Series], pd.DataFrame]
        The processed outputs in the given func_output. \n
        * If the `func_output` is a `pd.DataFrame`, the `pd.DataFrame` is returned.
        * If the `func_output` is a `pd.Series`, `np.ndarray`, or list of `pd.Series`,
          a series dict is returned.

    Raises
    ------
    TypeError
        Error raised when function output is invalid.

    Notes
    -----
    * If `func_output` is a `np.ndarray` or a `pd.Series` without a name, than the given
      `requested_dict` must contain just 1 series! That series its name and index are
      used to  transform (i.e., **replace**) that **series with the numpy array**.
      When a user does not want a numpy array to replace its input series, it is his /
      her responsibility to create a new `pd.Series` (or `pd.DataFrame`) of that numpy
      array with a different (column) name.
    * If `func_output` is a `pd.Series`, keep in mind that the input series gets
      transformed (i.e., **replaced**) with the `func_output` **when the series name is
      equal** or when there is no series name (see note above).

    """
    if isinstance(func_output, pd.DataFrame):
        # Nothing has to be done! A pd.DataFrame can be added to a series_dict using
        # series_dict.update(df)
        # Note: converting this to a dictionary (to_dict()) is **very** inefficient!
        # Assert that the DataFrame has a time-index
        if len(func_output):
            assert isinstance(func_output.index, pd.DatetimeIndex)
        # Assert that the DataFrame columns are named
        assert all(
            func_output.columns.values != [i for i in range(func_output.shape[1])]
        )
        return func_output

    elif isinstance(func_output, pd.Series):
        # Convert series to series_dict and return
        # => if func_output.name is in the required_dict, than the original series will
        #    be replaced by this new series.
        # Assert that the series has a time-index
        if len(func_output):
            assert isinstance(func_output.index, pd.DatetimeIndex)
        # Assert (func_output.name is not None) | (len(required_dict) == 1)
        if func_output.name is None:
            # If a series without a name is returned that is constructed from just 1
            # series => the input series will be replaced by this series
            assert len(required_dict) == 1
            # Rename, but keep the same reference to the data
            input_series = list(required_dict.values())[0]
            func_output.rename(input_series.name, inplace=True, copy=False)
        return {str(func_output.name): func_output}

    elif isinstance(func_output, np.ndarray):
        # Must be constructed from just 1 series
        # => the input series will be replaced by this array
        assert len(required_dict) == 1
        # Create a series with the same view
        input_series = list(required_dict.values())[0]
        return {str(input_series.name): _np_array_to_series(func_output, input_series)}

    elif isinstance(func_output, list) and all([isinstance(el, pd.Series) for el in func_output]):
        # Convert the list of series into a series dict, using the same data reference
        # => if for any series in the list, series.name is in the required_dict, than
        #    the the original series will be replaced by this new series.
        # Assert that all outputs have different names
        assert len(set([s.name for s in func_output])) == len(func_output)
        assert all([isinstance(s.index, pd.DatetimeIndex) for s in func_output])
        return {s.name: s for s in func_output}

    else:
        raise TypeError(f"Function output type is invalid for processor {func_name}")
