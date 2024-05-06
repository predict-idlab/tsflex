"""Utility functions for argumnt parsing (and time-related operations)."""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt"

import os
import warnings
from typing import Optional, Union

import pandas as pd


def parse_n_jobs(n_jobs: Optional[int]) -> int:
    """Parse the `n_jobs` argument.

    Parameters
    ----------
    n_jobs : Optional[int]
        The number of jobs to run in parallel.

    Returns
    -------
    int
        The number of jobs to run in parallel.
        If `n_jobs` is specified, it is returned.
        In the other case (i.e., `n_jobs` is `None`), the number of logical CPUs is
        returned (if available), otherwise a warning is raised and 1 is returned.

    """
    if n_jobs is not None:
        assert isinstance(n_jobs, int), "n_jobs must be an integer"
        return n_jobs
    # n_jobs is not specified
    _cpu_count = os.cpu_count()
    if _cpu_count is not None:
        n_jobs = _cpu_count
    else:
        warnings.warn(
            (
                "Number of logical CPUs is undetermined. Defaulting to 1. "
                + "To use more than 1 job, please specify the `n_jobs` argument."
            ),
            RuntimeWarning,
        )
        n_jobs = 1
    return n_jobs


def timedelta_to_str(td: pd.Timedelta) -> str:
    """Construct a tight string representation for the given timedelta arg.

    Parameters
    ----------
    td: pd.Timedelta
        The timedelta for which the string representation is constructed

    Returns
    -------
    str:
        The tight string bounds of format '$d-$h$m$s.$ms'.

    """
    out_str = ""

    # Edge case if we deal with negative
    if td < pd.Timedelta(seconds=0):
        td *= -1
        out_str += "NEG"

    # Note: this must happen after the *= -1
    c = td.components
    if c.days > 0:
        out_str += f"{c.days}D"
    if c.hours > 0 or c.minutes > 0 or c.seconds > 0 or c.milliseconds > 0:
        out_str += "_" if len(out_str) else ""

    if c.hours > 0:
        out_str += f"{c.hours}h"
    if c.minutes > 0:
        out_str += f"{c.minutes}m"
    if c.seconds > 0 or c.milliseconds > 0:
        out_str += f"{c.seconds}"
        if c.milliseconds:
            out_str += f".{str(c.milliseconds / 1000).split('.')[-1].rstrip('0')}"
        out_str += "s"
    return out_str


def parse_time_arg(arg: Union[str, pd.Timedelta]) -> pd.Timedelta:
    """Parse the `window`/`stride` arg into a fixed set of types.

    Parameters
    ----------
    arg : Union[float, str, pd.Timedelta]
        The arg that will be parsed. \n
        * If the type is a `pd.Timedelta`, nothing will happen.
        * If the type is a `str`, `arg` should represent a time-string, and will be
          converted to a `pd.Timedelta`.

    Returns
    -------
    pd.Timedelta
        The parsed time arg

    Raises
    ------
    TypeError
        Raised when `arg` is not an instance of `float`, `int`, `str`, or
        `pd.Timedelta`.

    """
    if isinstance(arg, pd.Timedelta):
        return arg
    elif isinstance(arg, str):
        if arg.isnumeric():
            raise ValueError(f"time-string arg {arg} must contain a unit")
        return pd.Timedelta(arg)
    raise TypeError(f"arg type {type(arg)} is not supported!")
