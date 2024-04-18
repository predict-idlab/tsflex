# TODO: rename file
"""(Advanced) utilities for the processing pipelines."""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt"

import traceback
from typing import Any, List, Optional, Union

import pandas as pd
from multiprocess import Pool
from tqdm.auto import tqdm

from ..utils.argument_parsing import parse_n_jobs
from .series_pipeline import SeriesPipeline


def process_chunks_multithreaded(  # type: ignore[no-untyped-def]
    same_range_chunks_list: List[List[Union[pd.Series, pd.DataFrame]]],
    series_pipeline: SeriesPipeline,
    show_progress: Optional[bool] = True,
    n_jobs: Optional[int] = None,
    **processing_kwargs,
) -> Optional[List[Any]]:
    """Process `same_range_chunks_list` in a multithreaded manner, order is preserved.

    Parameters
    ----------
    same_range_chunks_list: List[List[Union[pd.Series, pd.DataFrame]]]
        A list of same-range-chunks, most likely the output of `chunk_data`.
    series_pipeline: SeriesPipeline
        The pipeline that will process each item in `same_range_chunks_list`.
    show_progress: bool, optional
        If True, the progress will be shown with a progressbar, by default True.
    n_jobs: int, optional
        The number of processes used for the chunked series processing. If `None`, then
        the number returned by `os.cpu_count()` is used, by default None.
    **processing_kwargs
        Keyword arguments that will be passed on to the processing pipeline.

    Returns
    -------
    List[Any]
        A list of the `series_pipeline` its processed outputs. The order is preserved.

    Notes
    -----
    * This method is not concerned with joining the chunks as this operation is highly
      dependent on the preprocessing steps. This is the user's responsibility.
    * If any error occurs while executing the `series_pipeline` on one of the chunks
      in `same_range_chunks_list`, the traceback is printed and an empty dataframe is
      returned. We chose for this behavior, because in this way the other parallel
      processes are not halted in case of an error.

    """
    n_jobs = parse_n_jobs(n_jobs)

    def _executor(
        same_range_chunks: List[Union[pd.Series, pd.DataFrame]]
    ) -> Union[List[pd.Series], pd.DataFrame]:
        try:
            return series_pipeline.process(same_range_chunks, **processing_kwargs)
        except Exception:
            # Print traceback and return empty `pd.DataFrame` in order to not break the
            # other parallel processes.
            traceback.print_exc()
            return pd.DataFrame()

    processed_out = None
    with Pool(processes=min(n_jobs, len(same_range_chunks_list))) as pool:
        results = pool.imap(_executor, same_range_chunks_list)
        if show_progress:
            results = tqdm(results, total=len(same_range_chunks_list))
        try:
            processed_out = [f for f in results]
        except Exception:
            traceback.print_exc()
            pool.terminate()
        finally:
            # Close & join because: https://github.com/uqfoundation/pathos/issues/131
            pool.close()
            pool.join()
    return processed_out
