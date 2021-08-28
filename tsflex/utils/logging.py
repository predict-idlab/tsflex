"""Utility functions for logging operations."""

__author__ = 'Jeroen Van Der Donckt'

import logging
import warnings
import pandas as pd

from pathlib import Path
from typing import Union


def remove_inner_brackets(message: str) -> str:
    """Remove the inner brackets i.e., [ or ], from a string, outer brackets are kept.

    Parameters
    ----------
    message: str
        The string to remove the inner brackets from.

    Returns
    -------
    str:
        A new message without any inner brackets.
    
    """
    level = 0
    new_message = ""
    for char in message:
        if char == "[":
            if level == 0:
                new_message += char
            level += 1
        elif char == "]":
            if level == 1:
                new_message += char
            level -= 1
        else:
            new_message += char
        assert level >= 0
    return new_message


def delete_logging_handlers(logger: logging.Logger):
    """Delete all logging handlers that are not stream-handlers.

    Parameters
    ----------
    logger : logging.Logger
        The logger.
    
    """
    if len(logger.handlers) > 1:
        logger.handlers = [
            h for h in logger.handlers if type(h) == logging.StreamHandler
        ]
    assert len(logger.handlers) == 1, "Multiple logging StreamHandlers present!!"


def add_logging_handler(logger: logging.Logger, logging_file_path: Union[str, Path]):
    """Add a logging file-handler to the logger.

    Parameters
    ----------
    logger : logging.Logger
        The logger.
    logging_file_path : Union[str, Path]
        The file path for the file handler.
    
    """
    if not isinstance(logging_file_path, Path):
        logging_file_path = Path(logging_file_path)
    if logging_file_path.exists():
        warnings.warn(
            f"Logging file ({logging_file_path}) already exists. "
            f"This file will be overwritten!",
            RuntimeWarning,
        )
        # Clear the file
        #  -> because same FileHandler is used when calling this method twice
        open(logging_file_path, "w").close()
    f_handler = logging.FileHandler(logging_file_path, mode="w")
    f_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)


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
    df = pd.DataFrame(data)
    df["log_time"] = pd.to_datetime(df["log_time"])
    return df
