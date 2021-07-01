"""Tests for the tsflex.utils submodule."""

__author__ = "Jeroen Van Der Donckt, Emiel Deprost, Jonas Van Der Donckt"

import pandas as pd

from tsflex.utils.timedelta import timedelta_to_str


def test_timedelta_to_str():
    def to_td(td): return pd.Timedelta(td)

    # large time stamps
    assert timedelta_to_str(to_td('2W 1 day')) == '15D'

    # mediocre time-stamps
    assert timedelta_to_str(to_td('3days')) == '3D'
    assert timedelta_to_str(to_td('2hours')) == '2h'
    assert timedelta_to_str(to_td('2 hours')) == '2h'
    assert timedelta_to_str(to_td('10minutes')) == '10m'
    assert timedelta_to_str(to_td('10 minutes')) == '10m'
    assert timedelta_to_str(to_td('10min')) == '10m'
    assert timedelta_to_str(to_td('10 min')) == '10m'
    assert timedelta_to_str(to_td('1sec')) == '1s'

    # note! lower bound of timedelta_to_str = seconds
    assert timedelta_to_str(to_td('30milliseconds')) == '0.03s'
    assert timedelta_to_str(to_td('30 milliseconds')) == '0.03s'
    assert timedelta_to_str(to_td('1000microseconds')) == '0.001s'
    assert timedelta_to_str(to_td('1000000 nanoseconds')) == '0.001s'

    # negative timestamps
    assert timedelta_to_str(to_td('- 1 hour 30 minutes')) == 'NEG_1h30m'
    assert timedelta_to_str(to_td('- 2 hours')) == 'NEG_2h'
    assert timedelta_to_str(to_td('- 10minutes')) == 'NEG_10m'
    assert timedelta_to_str(to_td('- 10 minutes')) == 'NEG_10m'
    assert timedelta_to_str(to_td('- 10min')) == 'NEG_10m'
    assert timedelta_to_str(to_td('- 10 min')) == 'NEG_10m'
    assert timedelta_to_str(to_td('- 1sec')) == 'NEG_1s'

    # note! lower bound of timedelta_to_str = seconds
    assert timedelta_to_str(to_td('- 30milliseconds')) == 'NEG_0.03s'
    assert timedelta_to_str(to_td('- 30 milliseconds')) == 'NEG_0.03s'
    assert timedelta_to_str(to_td('- 1000microseconds')) == 'NEG_0.001s'
    assert timedelta_to_str(to_td('- 1000000 nanoseconds')) == 'NEG_0.001s'
