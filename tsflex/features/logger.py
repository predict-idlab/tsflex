"""Contains the used variables and functions to provide logging functionality.

See Also
--------
FeatureCollection: its `logging_file_path` of the `calculate` method.

"""

__author__ = "Jeroen Van Der Donckt"

import logging
import re

import numpy as np
import pandas as pd

from ..utils.argument_parsing import timedelta_to_str
from ..utils.logging import logging_file_to_df, remove_inner_brackets

# Package specific logger
logger = logging.getLogger("feature_calculation_logger")
logger.setLevel(logging.DEBUG)

# Create logger which writes WARNING messages or higher to sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logger.addHandler(console)


def _parse_message(message: str) -> list:
    """Parse the message of the logged info."""
    regex = r"\[(.*?)\]"
    matches = re.findall(regex, remove_inner_brackets(message))
    assert len(matches) == 5
    func = matches[0]
    key = matches[1].replace("'", "")
    window = matches[2].split(",")[0].strip()
    stride = ",".join(matches[2].split(",")[1:]).strip()
    if stride != "manual":
        stride = eval(stride)  # parse the tuple
    output_names = matches[3].replace("'", "")
    duration_s = float(matches[4].rstrip(" seconds"))
    return [func, key, window, stride, output_names, duration_s]


def _parse_logging_execution_to_df(logging_file_path: str) -> pd.DataFrame:
    """Parse the logged messages into a dataframe that contains execution info.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the `FeatureCollection` its `calculate` method.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the features its function, input series names, output names,
        and (%) calculation duration.

    Note
    ----
    This function only works when the ``logging_file_path`` used in a
    ``FeatureCollection`` its ``calculate`` method is passed.

    """
    df = logging_file_to_df(logging_file_path)
    df[
        ["function", "series_names", "window", "stride", "output_names", "duration"]
    ] = pd.DataFrame(
        list(df["message"].apply(_parse_message)),
        index=df.index,
    )
    # Parse the window
    if (df["window"] == "manual").any():
        # All should be manual
        assert (df["window"] == "manual").all()
    elif df["window"].str.isnumeric().all():
        df["window"] = pd.to_numeric(df["window"])
    else:
        df["window"] = pd.to_timedelta(df["window"]).apply(timedelta_to_str)
    # Parse the stride
    if (df["stride"] == "manual").any():
        # All should be manual
        assert (df["stride"] == "manual").all()
    elif (
        df["stride"]
        .apply(lambda stride_tuple: np.char.isnumeric(stride_tuple).all())
        .all()
    ):
        df["stride"] = df["stride"].apply(
            lambda stride_tuple: tuple(sorted(pd.to_numeric(s) for s in stride_tuple))
        )
    else:
        df["stride"] = df["stride"].apply(
            lambda stride_tuple: tuple(
                timedelta_to_str(pd.to_timedelta(s)) for s in stride_tuple
            )
        )
    df["duration %"] = (100 * (df["duration"] / df["duration"].sum())).round(2)
    return df.drop(columns=["name", "log_level", "message"])


def get_feature_logs(logging_file_path: str) -> pd.DataFrame:
    """Get execution (time) info for each feature of a `FeatureCollection`.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the `FeatureCollection` its `calculate` method.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the features its function, input series names and
        (%) calculation duration.

    """
    df = _parse_logging_execution_to_df(logging_file_path)
    df["duration"] = pd.to_timedelta(df["duration"], unit="s")
    return df


def get_function_stats(logging_file_path: str) -> pd.DataFrame:
    """Get execution (time) statistics for each function.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the `FeatureCollection` its `calculate` method.

    Returns
    -------
    pd.DataFrame
        A DataFrame with for each function (i.e., `function-(window,stride)`)
        combination the mean (time), std (time), sum (time), sum (% time),
        mean (% time),and number of executions.

    """
    df = _parse_logging_execution_to_df(logging_file_path)
    # Get the sorted functions in a list to use as key for sorting the groups
    sorted_funcs = (
        df.groupby(["function"])
        .agg({"duration": ["mean"]})
        .sort_values(by=("duration", "mean"), ascending=True)
        .index.to_list()
    )

    def key_func(idx_level):  # type: ignore[no-untyped-def]
        if all(idx in sorted_funcs for idx in idx_level):
            return [sorted_funcs.index(idx) for idx in idx_level]
        return idx_level

    return (
        df.groupby(["function", "window", "stride"])
        .agg(
            {
                "duration": ["sum", "mean", "std", "count"],
                "duration %": ["sum", "mean"],
            }
        )
        .sort_index(key=key_func, ascending=False)
    )


def get_series_names_stats(logging_file_path: str) -> pd.DataFrame:
    """Get execution (time) statistics for each `key-(window,stride)` combination.

    Parameters
    ----------
    logging_file_path: str
        The file path where the logged messages are stored. This is the file path that
        is passed to the `FeatureCollection` its `calculate` method.

    Returns
    -------
    pd.DataFrame
        A DataFrame with for each function the mean (time), std (time), sum (time),
        sum (% time), mean (% time), and number of executions.

    """
    df = _parse_logging_execution_to_df(logging_file_path)
    return (
        df.groupby(["series_names", "window", "stride"])
        .agg(
            {
                "duration": ["sum", "mean", "std", "count"],
                "duration %": ["sum", "mean"],
            }
        )
        .sort_values(by=("duration", "sum"), ascending=False)
    )
