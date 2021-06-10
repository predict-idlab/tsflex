"""Utility functions for timedelta operations."""

__author__ = 'Jonas Van Der Donckt'

import pandas as pd


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
        out_str += 'NEG'

    # Note: this must happen after the *= -1
    c = td.components
    if c.days > 0:
        out_str += f'{c.days}D'
    if c.hours > 0 or c.minutes > 0 or c.seconds > 0 or c.milliseconds > 0:
        out_str += '_' if len(out_str) else ""

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
