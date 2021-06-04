# -*- coding: utf-8 -*-
"""Various code utilities.
"""
__author__ = 'Jonas Van Der Donckt'

from datetime import timedelta
from typing import Dict, List, Union, Tuple, Optional

import pandas as pd


# TODO: maybe rename this method? -> to `timedelta_to_str`
def tightest_timedelta_bounds(td: pd.Timedelta) -> str:
    """Construct the tightest bounds string representation for the given timedelta arg.

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

    # edge case if we deal with negative
    if td < pd.Timedelta(seconds=0):
        td *= -1
        out_str += 'NEG'

    # note -> this must happen after the *= -1
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


def chunk_signals(
        signals: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
        fs_dict: Dict[str, int],
        verbose=False,
        cut_minute_wise: bool = False,
        chunk_range_margin_s: Optional[float] = None,
        min_chunk_dur_s: Optional[float] = None,
        max_chunk_dur_s: Optional[float] = None,
        sub_chunk_margin_s: Optional[float] = 0,
        copy=True,
) -> List[List[pd.Series]]:
    """Divide the `signals` in chunks.

    Does 2 things:

    1. Detecting gaps in the `signals`-list time series
    2. Divides the `signals` into chunks, according to the parameter
        configuration and the gaps.

    Note
    ----
    * As the `fs_dict` is requested, the assumption is made that **each signal** has a
      **fixed sample frequency**.
    * All subsequent signal-chunks are matched against the time-ranges of the first
      signal. This implies that **the first item in `signal` serves as a reference**
      for gap-matching.
    * The term `sub-chunk` refers to the chunks who exceed the `max_chunk_duration_s`
      parameter and are therefore further divided into sub-chunks.

    Parameters
    ----------
    signals : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
        The signals. Each signal must have a `DateTime-index`. The assumption is made
        that each signal has a _nearly_ fixed-sample frequency (when there are no gaps).
    fs_dict: Dict[str, int]
        The sample frequency dict. This dict must at least withhold all the keys
        from the `signals`.
    verbose : bool, optional
        If set, will print more verbose output, by default True
    cut_minute_wise : bool, optional
        If set, will cut on minute level granularity, by default False
    chunk_range_margin_s: float, optional
        The mar
    min_chunk_dur_s : float, optional
        The minimal duration of a chunk in seconds, by default None
        Chunks with durations smaller than this will not be processed.
    max_chunk_dur_s : float, optional
        The maximal duration of a chunk in seconds, by default None
        Chunks with durations larger than this will be chunked in smaller chunks where
        each sub-chunk has a maximal duration of `max_chunk_dur_s`.
    sub_chunk_margin_s: float, optional
        The left and right margin of the sub-chunks.
    copy: boolean, optional
        If set True will return a new view (on which you won't get a
        `SettingWithCopyWarning` if you change the content), by default False.

    Returns
    -------
    List[List[pd.Series]]
        A list of signals chunks.

    """
    # convert the input signals
    series_list: List[pd.Series] = []
    if not isinstance(signals, list):
        signals = [signals]

    for s in signals:
        if isinstance(s, pd.DataFrame):
            series_list += [s[c] for c in s.columns]
        elif isinstance(s, pd.Series):
            series_list.append(s)
        else:
            raise TypeError("Non pd.Series or pd.DataFrame object passed.")
    # assert that there are no duplicate signal names
    assert len(series_list) == len(set([s.name for s in series_list]))

    # Default arg -> set the chunk range margin to 2x the min-freq its period
    if chunk_range_margin_s is None:
        chunk_range_margin_s = 2 / min([fs_dict[str(s.name)] for s in series_list])
    chunk_range_margin_s = pd.Timedelta(seconds=chunk_range_margin_s)

    # variable in which the same time-range chunks are stored
    same_range_chunks: List[Tuple[pd.Timestamp, pd.Timestamp, List[pd.Series]]] = []

    def print_verbose_time(sig, t_begin, t_end, msg=""):
        fmt = "%Y-%m-%d %H:%M"
        if not verbose:
            return
        print(
            f"slice {sig.name} {t_begin.strftime(fmt):<10} - {t_end.strftime(fmt):<10} "
            f"-  shape: {sig[t_begin:t_end].shape}"
        )
        if len(msg):
            print(f"\t└──>  {msg}")

    def slice_time(sig: pd.Series, t_begin: pd.Timestamp,
                   t_end: pd.Timestamp) -> pd.Series:
        """Slice the ds_s dict."""
        if copy:
            return sig[t_begin:t_end].copy()
        else:
            return sig[t_begin:t_end]

    def insert_chunk(chunk: pd.Series):
        """Insert the chunk into the `df_list_dict`."""
        t_chunk_start, t_chunk_end = chunk.index[[0, -1]]

        # iterate over the same-(time)range-chunk (stc) collection
        for src_start, src_end, src_chunks in same_range_chunks:
            # check for overlap
            if (abs(src_start - t_chunk_start) < chunk_range_margin_s and
                    abs(src_end - t_chunk_end) < chunk_range_margin_s):
                # check signal name not in stc_chunks
                if chunk.name not in [src_c.name for src_c in src_chunks]:
                    src_chunks.append(chunk)
                    print_verbose_time(chunk, t_chunk_start, t_chunk_end,
                                       "INSERT chunk")
                    return
                else:
                    # There already exists a sub_chunk of this signal-name
                    raise ValueError("There already exists a chunk with this signal"
                                     f" name - {chunk.name}")

        same_range_chunks.append((t_chunk_start, t_chunk_end, [chunk]))
        print_verbose_time(chunk, t_chunk_start, t_chunk_end, "APPEND sub chunk")

    for signal in series_list:
        # 1. Some checks
        if len(signal) < 2:
            if verbose:
                print(f"too small signal: {signal.name} - shape: {signal.shape} ")
            continue
        assert signal.name in fs_dict

        fs_sensor = fs_dict[str(signal.name)]

        # Allowed offset (in seconds) is sample_period + 0.5*sample_period
        gaps = signal.index.to_series().diff() > timedelta(seconds=(1 + .5) / fs_sensor)
        # Set the first and last timestamp to True
        gaps.iloc[[0, -1]] = True
        gaps: List[pd.Timestamp] = signal[gaps].index.to_list()
        if verbose:
            print("-" * 10, " detected gaps", "-" * 10)
            print(*gaps, sep="\n")

        # Reset the iterator
        for (t_begin_c, t_end_c) in zip(gaps, gaps[1:]):
            if cut_minute_wise:
                t_begin_c = (t_begin_c + timedelta(seconds=60)).replace(
                    second=0, microsecond=0
                )

                # As we add time -> we might want to add this sanity check
                if t_begin_c > t_end_c:
                    print_verbose_time(
                        signal, t_begin_c, t_end_c, "[W] t_end > t_start"
                    )
                    continue

            # The t_end is the t_start of the new time range -> hence [:-1]
            # => cut on [t_start_c(hunk), t_end_c(hunk)[
            sig_chunk = signal[t_begin_c:t_end_c][:-1]
            if len(sig_chunk) > 2:  # re-adjust the t_end
                t_end_c = sig_chunk.index[-1]
            else:
                print_verbose_time(signal, t_begin_c, t_end_c, "too small df_chunk")
                continue

            # Check for min duration
            chunk_range_s = len(sig_chunk) // fs_sensor
            if isinstance(min_chunk_dur_s, int) and chunk_range_s < min_chunk_dur_s:
                print_verbose_time(
                    sig_chunk,
                    t_begin_c,
                    t_end_c,
                    f"Too small chunk min_dur {min_chunk_dur_s} > {sig_chunk.shape}",
                )
                continue

            # Divide the chunk into sub_chunks (sc's)
            if max_chunk_dur_s is not None and chunk_range_s > max_chunk_dur_s:
                print_verbose_time(
                    sig_chunk, t_begin_c, t_end_c, "Dividing in sub-chunks"
                )
                t_begin_sc = t_begin_c
                while t_begin_sc < t_end_c:
                    # Slice, by making use of the margin
                    t_end_sc = t_begin_sc + timedelta(seconds=max_chunk_dur_s)
                    t_end_sc_m = t_end_sc + timedelta(seconds=sub_chunk_margin_s)
                    t_end_sc_m = min(t_end_c, t_end_sc_m)

                    t_begin_sc_m = t_begin_sc - timedelta(seconds=sub_chunk_margin_s)
                    t_begin_sc_m = max(t_begin_c, t_begin_sc_m)

                    # Slice & add the sub-chunk to the list
                    insert_chunk(chunk=slice_time(signal, t_begin_sc_m, t_end_sc_m), )

                    # Update the condition's variable
                    t_begin_sc = t_end_sc
            else:
                insert_chunk(chunk=slice_time(signal, t_begin_c, t_end_c))
    return [chunk_range[-1] for chunk_range in same_range_chunks]
