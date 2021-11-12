# -*- coding: utf-8 -*-
"""(Advanced) tsflex utilities for chunking sequence data."""

__author__ = "Jonas Van Der Donckt"

from datetime import timedelta
from typing import Dict, List, Union, Tuple, Optional

import pandas as pd

from ..utils.attribute_parsing import AttributeParser, DataType
from ..utils.data import to_series_list
from ..utils.time import parse_time_arg


def _chunk_time_data(
    series_list: List[pd.Series],
    fs_dict: Optional[Dict[str, float]] = None,
    chunk_range_margin: Optional[Union[str, pd.Timedelta]] = None,
    min_chunk_dur: Optional[Union[str, pd.Timedelta]] = None,
    max_chunk_dur: Optional[Union[str, pd.Timedelta]] = None,
    sub_chunk_overlap: Optional[Union[str, pd.Timedelta]] = "0s",
    copy=True,
    verbose=False,
):
    if min_chunk_dur is not None:
        min_chunk_dur = parse_time_arg(min_chunk_dur)
    if max_chunk_dur is not None:
        max_chunk_dur = parse_time_arg(max_chunk_dur)
    sub_chunk_overlap = parse_time_arg(sub_chunk_overlap)

    # Default arg -> set the chunk range margin to 2x the min-freq its period
    if chunk_range_margin is None:
        if fs_dict is not None:
            chunk_range_margin = pd.Timedelta(
                seconds=2 / min([fs_dict[str(s.name)] for s in series_list])
            )
        else:
            raise ValueError("Chunk range margin must be set if fs_dict is not set!")

    chunk_range_margin = parse_time_arg(chunk_range_margin)
    assert chunk_range_margin.total_seconds() > 0, "chunk_range_margin must be > 0"

    # if fs_dict is not set -> set it to the max time-diff for the corresponding series
    if fs_dict is None:
        if verbose:
            print("fs is none -> using 1 / max time diff for each series as fs")
        fs_dict = {
            s.name: (1 / s.index.to_series().diff().max().total_seconds())
            for s in series_list
        }

    # Assert the names reside in fs_dict
    assert all([str(s.name) in fs_dict for s in series_list])

    # Some range asserts
    assert sub_chunk_overlap.total_seconds() >= 0, f"sub_chunk_overlap_s must be > 0"
    if max_chunk_dur is not None:
        assert max_chunk_dur.total_seconds() > 0, f"max_chunk_dur_ must be > 0"

    # Variable in which the same time-range chunks are stored
    # Each list item can be seen as (t_start_chunk, t_end_chunk, chunk_list)
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

    def slice_time(
        sig: pd.Series, t_begin: pd.Timestamp, t_end: pd.Timestamp
    ) -> pd.Series:
        """Slice the sig Series."""
        if copy:
            return sig[t_begin:t_end].copy()
        else:
            return sig[t_begin:t_end]

    def insert_chunk(chunk: pd.Series):
        """Insert the chunk into `same_range_chunks`."""
        t_chunk_start, t_chunk_end = chunk.index[[0, -1]]

        # Iterate over the same-(time)range-chunk (src) collection
        for src_start, src_end, src_chunks in same_range_chunks:
            # Check for overlap
            if (
                abs(src_start - t_chunk_start) <= chunk_range_margin
                and abs(src_end - t_chunk_end) <= chunk_range_margin
            ):
                # Check series name not in src_chunks
                if chunk.name not in [src_c.name for src_c in src_chunks]:
                    src_chunks.append(chunk)
                    print_verbose_time(
                        chunk, t_chunk_start, t_chunk_end, "INSERT chunk"
                    )
                    return
                else:
                    # There already exists a sub_chunk of this series name
                    raise ValueError(
                        "There already exists a chunk with this series"
                        f" name - {chunk.name}"
                    )

        same_range_chunks.append((t_chunk_start, t_chunk_end, [chunk]))
        print_verbose_time(chunk, t_chunk_start, t_chunk_end, "APPEND sub chunk")

    for series in series_list:
        # 1. Some checks
        if len(series) < 2:
            if verbose:
                print(f"Too small series: {series.name} - shape: {series.shape} ")
            continue

        # Allowed offset (in seconds) is sample_period + 0.5*sample_period
        fs_sig = fs_dict[str(series.name)]
        gaps = series.index.to_series().diff() > timedelta(seconds=(1 + 0.5) / fs_sig)
        # Set the first and last timestamp to True
        gaps.iloc[[0, -1]] = True
        gaps: List[pd.Timestamp] = series[gaps].index.to_list()
        if verbose:
            print("-" * 10, " detected gaps", "-" * 10)
            print(*gaps, sep="\n")

        for (t_begin_c, t_end_c) in zip(gaps, gaps[1:]):
            # The t_end is the t_start of the new time range -> hence [:-1]
            # => cut on [t_start_c(hunk), t_end_c(hunk)[
            sig_chunk = series[t_begin_c:t_end_c]
            if t_end_c < gaps[-1]:
                # Note: we doe not adjust when t_end_c = gaps[-1]
                #   (as gaps-[-1] is not really a gap)
                sig_chunk = sig_chunk[:-1]

            if len(sig_chunk) > 2:  # re-adjust the t_end
                t_end_c = sig_chunk.index[-1]
            else:
                print_verbose_time(series, t_begin_c, t_end_c, "too small df_chunk")
                continue

            # Check for min duration
            chunk_range_s = int(len(sig_chunk) / fs_sig)
            if (
                min_chunk_dur is not None
                and chunk_range_s < min_chunk_dur.total_seconds()
            ):
                print_verbose_time(
                    sig_chunk,
                    t_begin_c,
                    t_end_c,
                    f"Too small chunk min_dur {min_chunk_dur} > {sig_chunk.shape}",
                )
                continue

            # Divide the chunk into sub_chunks (sc's)
            if (
                max_chunk_dur is not None
                and chunk_range_s > max_chunk_dur.total_seconds()
            ):
                print_verbose_time(
                    sig_chunk, t_begin_c, t_end_c, "Dividing in sub-chunks"
                )
                t_begin_sc = t_begin_c
                while t_begin_sc < t_end_c:
                    # Calculate the end sub-chunk time
                    t_end_sc = t_begin_sc + max_chunk_dur

                    # Get the end and begin sub-chunk margin
                    t_end_sc_m = t_end_sc + sub_chunk_overlap / 2
                    t_end_sc_m = min(t_end_c, t_end_sc_m)

                    t_begin_sc_m = t_begin_sc - sub_chunk_overlap / 2
                    t_begin_sc_m = max(t_begin_c, t_begin_sc_m)

                    # Slice & add the sub-chunk to the list
                    insert_chunk(chunk=slice_time(series, t_begin_sc_m, t_end_sc_m))

                    # Update the condition's variable
                    t_begin_sc = t_end_sc
            else:
                insert_chunk(chunk=slice_time(series, t_begin_c, t_end_c))
    return [chunk_range[-1] for chunk_range in same_range_chunks]


def _chunk_sequence_data(
    series_list: List[pd.Series],
    fs_dict: Optional[Dict[str, float]] = None,
    chunk_range_margin: Optional[float] = None,
    min_chunk_dur: Optional[float] = None,
    max_chunk_dur: Optional[float] = None,
    sub_chunk_overlap: Optional[float] = "0s",
    copy=True,
    verbose=False,
):
    raise NotImplementedError("Not implemented yet")


_dtype_to_chunk_method = {
    DataType.TIME: _chunk_time_data,
    DataType.SEQUENCE: _chunk_sequence_data,
}


def chunk_data(
    data: Union[
        pd.Series,
        pd.DataFrame,
        List[Union[pd.Series, pd.DataFrame]],
        Dict[str, pd.DataFrame],
    ],
    fs_dict: Optional[Dict[str, float]] = None,
    chunk_range_margin: Optional[Union[float, str, pd.Timedelta]] = None,
    min_chunk_dur: Optional[Union[float, str, pd.Timedelta]] = None,
    max_chunk_dur: Optional[Union[float, str, pd.Timedelta]] = None,
    sub_chunk_overlap: Optional[Union[float, str, pd.Timedelta]] = "0s",
    copy=True,
    verbose=False,
) -> List[List[pd.Series]]:
    """Divide the time-series `data` in same time/sequence-range chunks.

    Does 2 things:

    1. Detecting gaps in the `data`(-list) sequence series.
    2. Divides the `data` into chunks, according to the parameter
       configuration and the detected gaps.

    Notes
    -----
    * When you set `fs_dict`, the assumption is made that **each item** in `data`
      has a **fixed sample frequency**. If you do not set `fs_dict`, this variable
      will use the 1 / max time-diff of the corresponding series as key-value pair.
    * All subsequent series-chunks are matched against the time-ranges of the first
      series. This implies that **the first item in `data` serves as a reference**
      for gap-matching.
    * The term `sub-chunk` refers to the chunks who exceed the `max_chunk_duration`
      parameter and are therefore further divided into sub-chunks.

    Example
    -------
    ```python
    df_acc  # cols ['ACC_x', 'ACC_y`, 'ACC_z`, 'ACC_SMV`] - 32 Hz
    df_gyro # cols ['gyro_x', 'gyro_y`, 'gyro_z`, 'gyro_area`] - 100 Hz
    chunk_data({'acc': df_acc, 'g': df_gyro}, fs_dict={'acc': 32, 'g': 100})
    ```
    <br>

    .. Note::
        If `chunk_range_margin` / `min_chunk_dur` / `max_chunk_dur` / 
        `sub_chunk_overlap` is a int/float, it will be interpreted as numerical
        sequence range  and a numerical-indexed `data` will be assumed.
        **These attributes must be all either time-based or numerical and match
        the data its index dtype**

    Parameters
    -----------
    data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]], Dict[str, pd.DataFrame]]
        The sequence data which will be chunked. Each item in `data` must have a
        monotonically increasing index. We assume that each `item` in data
        has a _nearly-constant_ sample frequency (when there are no gaps) and all 
        indices have the same dtype.
    fs_dict: Dict[str, int], optional
        The sample frequency dict. If set, this dict must at least withhold all the keys
        from the items in `data`.
        .. note::
            if you passed a **_DataFrame-dict_** (i.e., a dict with key=str;
            value=DataFrame) to `data`, then you can **use** the **corresponding
            dataframe str-key** to describe the `fs` for all the DataFrame its columns
            with the `fs_dict` attribute. See also the example above
    chunk_range_margin: Union[float, str, pd.Timedelta], optional
        The allowed margin for each `ts` chunk their start and end time to be seen as 
        same time-range chunks with other `ts`. If `None` the margin will be set as:

            2 / min(fs_dict.intersection(data.names).values())

        Which is equivalent to twice the min-fs (= max-period) of the passed `data`,
        by default None.\n
        * if `pd.Timedelta`, it will be interpreted as a time-range margin
        * if `int` or `float`, it will be interpreted as a numerical range margin
    min_chunk_dur : Union[float, str, pd.Timedelta], optional
        The minimum duration of a chunk, by default None.
        Chunks with durations smaller than this will be discarded (and not returned).\n
        * if `pd.Timedelta`, it will be interpreted as a time-range margin
        * if `int` or `float`, it will be interpreted as a numerical range margin
    max_chunk_dur : Union[float, str, pd.Timedelta], optional
        The maximum duration of a chunk, by default None.
        Chunks with durations larger than this will be chunked in smaller `sub_chunks`
        where each sub-chunk has a maximum duration of `max_chunk_dur`.\n
        * if `pd.Timedelta`, it will be interpreted as a time-range margin
        * if `int` or `float`, it will be interpreted as a numerical range margin
    sub_chunk_overlap: Union[float, str, pd.Timedelta], optional
        The sub-chunk boundary overlap. If available, **this
        margin / 2 will be added to either side of the `sub_chunk`**.
        This is especially useful to not lose inter-`sub_chunk` data (as each
        `sub_chunk` is in fact a continuous chunk) when window-based aggregations
        are performed on these same time range output (sub_)chunks.
        This argument is only relevant if `max_chunk_dur` is set.\n
        * if `pd.Timedelta`, it will be interpreted as a time-range margin
        * if `int` or `float`, it will be interpreted as a numerical range margin
    copy: boolean, optional
        If set True will return a new view (on which you won't get a
        `SettingWithCopyWarning` if you change the content), by default False.
    verbose : bool, optional
        If set, will print more verbose output, by default False

    Returns
    -------
    List[List[pd.Series]]
        A list of same time range chunks.

    """
    if isinstance(data, dict):
        if isinstance(fs_dict, dict):
            out_dict = {}
            for k, fs in fs_dict.items():
                if k in data and isinstance(data[k], pd.DataFrame):
                    out_dict.update({c_name: fs for c_name in data[k].columns})
            fs_dict.update(out_dict)

        # make `data` `to_series_list` convertable()
        data = list(data.values())

    # Convert the input data
    series_list = to_series_list(data)

    # Assert that there are no duplicate series names
    assert len(series_list) == len(set([s.name for s in series_list]))

    # Assert that the index increases monotonically
    assert all(s.index.is_monotonic_increasing for s in series_list)

    return _dtype_to_chunk_method[AttributeParser.determine_type(data)](
        series_list,
        fs_dict,
        chunk_range_margin,
        min_chunk_dur,
        max_chunk_dur,
        sub_chunk_overlap,
        copy,
        verbose,
    )
