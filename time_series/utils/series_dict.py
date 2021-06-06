"""Utility functions for series dicts (the internal representation used inside tsflex)."""

__author__ = 'Jeroen Van Der Donckt'

from typing import Dict

import pandas as pd

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
    The `series_dict` is an internal representation of the signals list.
    In this dictionary, the key is always the accompanying series its name.
    This internal representation is constructed in the `process` method of the
    `SeriesPipeline`.

    """
    # 0. Check if the series_dict has only 1 signal, to create the df efficiently
    if len(series_dict) == 1:
        return pd.DataFrame(series_dict)
    # 1. Check if the time-indexes of the series are equal, to create the df efficiently
    try:
        index_info = set(
            [(s.index[0], s.index[-1], len(s), s.index.freq) for s in series_dict.values()]
        )
        if len(index_info) == 1:
            # If list(index_info)[0][-1] is None => this code assumes equal index to
            # perform efficient merge, otherwise the join will be still correct, but it
            # would actually be more efficient to perform the code at (2.).
            # But this disadvantage (slower merge) does not outweigh the time-loss when
            # checking the full index.

            # When the time-indexes are the same we can create df very efficiently
            return pd.DataFrame(series_dict)
    except IndexError:
        # We catch an indexError as we make the assumption that there is data within the
        # series -> we do not make that assumption when constructing the DataFrame the
        # slow way.
        pass
    # 2. If check failed, create the df by merging the series (the slow way)
    df = pd.DataFrame()
    for key, s in series_dict.values():
        # Check if we deal with a valid series_dict before merging on series.name
        assert key == s.name  
        df = df.merge(s, left_index=True, right_index=True, how="outer")
    return df
