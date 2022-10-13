# -*- coding: utf-8 -*-
"""
Factory class for creating the proper StridedRolling instances.

.. TODO::
    Also create a (SegmenterFactory) which the StridedRollingFactory implements

"""

__author__ = "Jonas Van Der Donckt"

from tracemalloc import start

import numpy as np

from ...utils.attribute_parsing import AttributeParser, DataType
from .strided_rolling import (
    SequenceStridedRolling,
    StridedRolling,
    TimeIndexSampleStridedRolling,
    TimeStridedRolling,
)


class StridedRollingFactory:
    """Factory class for creating the appropriate StridedRolling segmenter."""

    _datatype_to_stroll = {
        DataType.TIME: TimeStridedRolling,
        DataType.SEQUENCE: SequenceStridedRolling,
    }

    @staticmethod
    def get_segmenter(data, window, strides, **kwargs) -> StridedRolling:
        """Get the appropriate StridedRolling instance for the passed data.

        The returned instance will be determined by the data its index type

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            The data to segment.
        window : Union[float, pd.TimeDelta]
             The window size to use for the segmentation.
        strides : Union[List[Union[float, pd.TimeDelta]], None]
            The stride(s) to use for the segmentation.
        **kwargs : dict, optional
            Additional keyword arguments, see the `StridedRolling` its documentation
            for more info.

        .. Note::
            When passing `time-based` data, but **int**-based window & stride params,
            the strided rolling will be `TimeIndexSampleStridedRolling`. This class
            **assumes** that **all data series** _roughly_ have the
            **same sample frequency**, as  the windows and strides are interpreted in
            terms of **number of samples**.

        Raises
        ------
        ValueError
            When incompatible segment_indices, data & window-stride data types are
            passed (e.g. time window-stride args on sequence data-index).

        Returns
        -------
        StridedRolling
            The constructed StridedRolling instance.

        """
        data_dtype = AttributeParser.determine_type(data)

        # Get the start and end indices of the data and replace them with [] when None
        start_indices = kwargs.get("segment_start_idxs")
        # start_indices = [] if start_indices is None else start_indices
        end_indices = kwargs.get("segment_end_idxs")
        # end_indices = [] if end_indices is None else end_indices

        if strides is None:
            ws_dtype = AttributeParser.determine_type(window)
        else:
            ws_dtype = AttributeParser.determine_type([window] + strides)

        if isinstance(start_indices, np.ndarray) and isinstance(
            end_indices, np.ndarray
        ):
            # When both segment_indices are passed, this must match the data dtype
            segment_dtype = AttributeParser.determine_type(start_indices)
            assert segment_dtype == AttributeParser.determine_type(end_indices)
            if segment_dtype != DataType.UNDEFINED:
                assert segment_dtype == data_dtype, (
                    "Currently, only TimeStridedRolling and SequenceStridedRolling are "
                    + "supported, as such, the segment and data dtype must match;"
                    + f"Got seg_dtype={segment_dtype} and data_dtype={data_dtype}."
                )
                window = None
                return StridedRollingFactory._datatype_to_stroll[segment_dtype](
                    data, window, strides, **kwargs
                )
        elif isinstance(start_indices, np.ndarray) or isinstance(
            end_indices, np.ndarray
        ):
            # if only one of the start and end-indices are passed, we must check
            # if these are compatible with the window and stride params
            segment_dtype = AttributeParser.determine_type(
                start_indices if start_indices is not None else end_indices
            )
            assert segment_dtype == ws_dtype, (
                f"Segment start/end indices must be of the same type as the window "
                + "and stride params when only one of the two segment indices is given."
                + f"Got seg_dtype={segment_dtype} and ws_dtype={ws_dtype}."
            )

        if window is None or data_dtype.value == ws_dtype.value:
            return StridedRollingFactory._datatype_to_stroll[data_dtype](
                data, window, strides, **kwargs
            )
        elif data_dtype == DataType.TIME and ws_dtype == DataType.SEQUENCE:
            # Note: this is very niche and thus requires advanced knowledge
            return TimeIndexSampleStridedRolling(data, window, strides, **kwargs)
        elif data_dtype == DataType.SEQUENCE and ws_dtype == DataType.TIME:
            raise ValueError("Cannot segment a sequence-series with a time window")
