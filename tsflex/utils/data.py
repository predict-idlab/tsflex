"""Utility functions for internal data (representation) operations."""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt"

import itertools
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd

SUPPORTED_STROLL_TYPES = [np.ndarray, pd.Series]


def series_dict_to_df(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
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
    The `series_dict` is an internal representation of the time-series data.
    In this dictionary, the key is always the accompanying series its name.
    This internal representation is constructed in the `process` method of the
    `SeriesPipeline`.

    .. todo::
        @Jeroen - check performance

    """
    # 0. Check if the series_dict has only 1 series, to create the df efficiently
    if len(series_dict) == 1:
        return pd.DataFrame(series_dict)
    # 1. Check if the time-indexes of the series are equal, to create the df efficiently
    try:

        def _get_index_freq(index):  # type: ignore[no-untyped-def]
            if isinstance(index, pd.DatetimeIndex):
                return index.freq
            elif isinstance(index, pd.RangeIndex):
                return index.step
            return None

        index_info = set(
            [
                (s.index[0], s.index[-1], len(s), _get_index_freq(s.index))
                for s in series_dict.values()
            ]
        )
        if len(index_info) == 1:
            # If list(index_info)[0][-1] is None => this code assumes equal index to
            # perform efficient merge, otherwise the join will be still correct, but it
            # would actually be more efficient to perform the code at (2.).
            # But this disadvantage (slower merge) does not outweigh the time-loss when
            # checking the full index.

            # When the time-indexes are the same we can create df very efficiently
            return pd.DataFrame(series_dict, copy=False)
    except IndexError:
        # We catch an indexError as we make the assumption that there is data within the
        # series -> we do not make that assumption when constructing the DataFrame the
        # slow way.
        pass
    # 2. If check failed, create the df by merging the series (the slow way)
    df = pd.DataFrame()
    for key, s in series_dict.items():
        # Check if we deal with a valid series_dict before merging on series.name
        assert key == s.name
        df = df.merge(s, left_index=True, right_index=True, how="outer", copy=False)
    return df


def to_series_list(
    data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
) -> List[pd.Series]:
    """Convert the data to a list of series.

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]
        Dataframe or Series or list thereof, that should be transformed to a series
        list.

    Returns
    -------
    List[pd.Series]
        List of series containing the series present in `data`.

    """
    if not isinstance(data, list):
        data = [data]

    series_list: List[pd.Series] = []
    for s in data:
        if isinstance(s, pd.DataFrame):
            series_list += [s[c] for c in s.columns]
        elif isinstance(s, pd.Series):
            series_list.append(s)
        else:
            raise TypeError("Non pd.Series or pd.DataFrame object passed.")

    return series_list


def to_list(x: Any) -> List:
    """Convert the input to a list if necessary.

    Parameters
    ----------
    x : Any
        The input that needs to be converted into a list.

    Returns
    -------
    List
        A list of `x` if `x` wasn't a list yet, otherwise `x`.

    """
    if not isinstance(x, (list, np.ndarray)):
        return [x]
    return list(x)


def to_tuple(x: Any) -> Tuple[Any, ...]:
    """Convert the input to a tuple if necessary.

    Parameters
    ----------
    x : Any
        The input that needs to be converted into a tuple.

    Returns
    -------
    List
        A tuple of `x` if `x` wasn't a tuple yet, otherwise `x`.

    """
    if not isinstance(x, tuple):
        return (x,)
    return x


def flatten(data: Iterable) -> Iterator:
    """Flatten the given input data to an iterator.

    Parameters
    ----------
    data : Iterable
        The iterable data that needs to be flattened.

    Returns
    -------
    Iterator
        An iterator for the flattened data.

    """
    return itertools.chain.from_iterable(data)


def load_empatica_data(f_names: Union[str, List[str]]) -> List[pd.DataFrame]:
    """load example empatica data from the GitHub repository.

    Parameters
    ----------
    f_names : List[str]k
        The name-prefixes of the files, must be one or more from

            ["acc", "gsr", "tmp", "ibi"]

    Returns
    -------
    List[pd.DataFrame]
        Returns the empatica time-indexed data files in the same order as `f_names`
    """
    empatica_dir: Path = (
        Path(__file__)
        .parent.parent.parent.joinpath("examples", "data", "empatica")
        .absolute()
    )
    empatica_dir_str: str = (
        str(empatica_dir) + "/"
    )  # allows compatible + operation (as with the url)
    url = "https://github.com/predict-idlab/tsflex/raw/main/examples/data/empatica/"
    if not os.path.exists(empatica_dir_str):
        empatica_dir_str = url  # fetch online if data not local
    f_names = [f_names] if isinstance(f_names, str) else f_names
    return [
        pd.read_parquet(empatica_dir_str + f"{f_name.lower()}.parquet").set_index(
            "timestamp"
        )
        for f_name in f_names
    ]
