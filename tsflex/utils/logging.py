"""Utility functions for logging operations."""

__author__ = 'Jeroen Van Der Donckt'

import pandas as pd


def logging_file_to_df(logging_file_path: str) -> pd.DataFrame:
    """Parse the logged messages into a dataframe.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the log_time, name, log_level and log message.

    """
    column_names = ["log_time", "name", "log_level", "message"]
    data = {col: [] for col in column_names}
    with open(logging_file_path, "r") as f:
        for line in f:
            line = line.split(" - ")
            for idx, col in enumerate(column_names):
                data[col].append(line[idx].strip())
    return pd.DataFrame(data)
