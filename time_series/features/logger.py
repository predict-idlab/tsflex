"""Contains the used variables and methods to provide logging functionality.

See Also
--------
The `FeatureCollection` its `logging_file_path` of the calculation method.

"""

__author__ = "Jeroen Van Der Donckt"

import logging
from typing import Union, Tuple
import pandas as pd
import re

logger = logging.getLogger("feature_calculation_logger")
logger.setLevel(logging.DEBUG)

# Create logger which writes WARNING messages or higher to sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logger.addHandler(console)


def _parse_message(message: str) -> list:
    """Parse the message of the logged info."""
    regex = "\[(.*?)\]"
    matches = re.findall(regex, message)
    assert len(matches) == 3
    func = matches[0]
    key = matches[1].strip("'")  # TODO: up until now a function can have just 1 key?
    duration = float(matches[2].rstrip(" seconds"))
    return [func, key, duration]


def parse_logging_execution_to_df(logging_file_path: str) -> pd.DataFrame:
    """Parse the logged messages into a dataframe that contains execution info.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the feature collection its `calculate` method. # TODO: update if calculate gets replaced

    Returns
    pd.DataFrame
        A DataFrame with the features its method, input keys and calculation duration.

    """
    column_names = ["log_time", "name", "log_level", "message"]
    data = {col: [] for col in column_names}
    with open(logging_file_path, "r") as f:
        for line in f:
            line = line.split(" - ")
            for idx, col in enumerate(column_names):
                data[col].append(line[idx].strip())
    df = pd.DataFrame(data)
    df[["function", "key", "duration"]] = list(df["message"].apply(_parse_message))
    return df.drop(columns=["name", "log_level", "message"])


def get_function_duration_stats(logging_file_path: str) -> pd.DataFrame:
    """Get execution (time) statistics for each function.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the feature collection its `calculate` method. # TODO: update if calculate gets replaced

    Returns
    pd.DataFrame
        A DataFrame with for each function the mean (time), std (time), sum (time), and
        number of executions.

    """
    df = parse_logging_execution_to_df(logging_file_path)
    return (
        df.groupby("function")
        .agg({"duration": ["mean", "std", "sum", "count"]})
        .sort_values(by=("duration", "mean"), ascending=False)
    )


def get_key_duration_stats(logging_file_path: str) -> pd.DataFrame:
    """Get execution (time) statistics for each .

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the feature collection its `calculate` method. # TODO: update if calculate gets replaced

    Returns
    pd.DataFrame
        A DataFrame with for each function the mean (time), std (time), sum (time), and
        number of executions.

    """
    df = parse_logging_execution_to_df(logging_file_path)
    return (
        df.groupby("key")
        .agg({"duration": ["sum", "mean", "std", "count"]})
        .sort_values(by=("duration", "sum"), ascending=False)
    )
