"""Contains the used variables and methods to provide logging functionality.

See Also
--------
The `SeriesProcessorPipeline` its `logging_file_path` of the call method.

"""

__author__ = "Jeroen Van Der Donckt"

import logging
import pandas as pd
import re

# Package specific logger
logger = logging.getLogger("feature_processing_logger")
logger.setLevel(logging.DEBUG)

# Create logger which writes WARNING messages or higher to sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logger.addHandler(console)


def _parse_message(message: str) -> list:
    """Parse the message of the logged info."""
    regex = "\[(.*?)\]"
    matches = re.findall(regex, message)
    assert len(matches) == 4
    func = matches[0]
    single_series_func = bool(matches[1].lstrip("single_series_func="))
    keys = matches[2].strip("'")
    duration_s = float(matches[3].rstrip(" seconds"))
    return [func, single_series_func, keys, duration_s]


def parse_logging_execution_to_df(logging_file_path: str) -> pd.DataFrame:
    """Parse the logged messages into a dataframe that contains execution info.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the SeriesProcessorPipeline its `__call__` method.

    Note
    ----
    This function only works when the `logging_file_path` that is used in a 
    SeriesProcessorPipeline is passed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the processor its method, keys and calculation duration.

    """
    column_names = ["log_time", "name", "log_level", "message"]
    data = {col: [] for col in column_names}
    with open(logging_file_path, "r") as f:
        for line in f:
            line = line.split(" - ")
            for idx, col in enumerate(column_names):
                data[col].append(line[idx].strip())
    df = pd.DataFrame(data)
    df[["function", "single_series_func", "keys", "duration"]] = list(df["message"].apply(_parse_message))
    return df.drop(columns=["name", "log_level", "message"])


def get_duration_stats(logging_file_path: str) -> pd.DataFrame:
    """Get execution (time) statistics for each function of a SeriesProcessorPipeline.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the SeriesProcessorPipeline its `__call___` method.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing each function its duration, required keys, and whether it 
        is a `single_series_func` .

    """
    df = parse_logging_execution_to_df(logging_file_path)
    return df
