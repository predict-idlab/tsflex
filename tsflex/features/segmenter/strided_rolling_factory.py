# -*- coding: utf-8 -*-
"""
Factory class for creating the proper StridedRolling instances.

.. TODO::
    Maybe also create a (SegmenterFactory) which the StridedRollingFactory implements

"""
__author__ = "Jonas Van Der Donckt"

from .strided_rolling import (
    StridedRolling,
    TimeStridedRolling,
    SequenceStridedRolling,
    TimeIndexSequenceStridedRolling,
)
from ...utils.attribute_parsing import AttributeParser, DataType


class StridedRollingFactory:
    _datatype_to_stroll = {
        DataType.TIME: TimeStridedRolling,
        DataType.SEQUENCE: SequenceStridedRolling,
    }

    @staticmethod
    def get_segmenter(data, window, stride, **kwargs) -> StridedRolling:
        """Get the appropriate StridedRolling instance for the passed data.

        The returned instance will be determined by the data its index type

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            The data to segment.
        window: Union[float, pd.TimeDelta]
             Either an int, float, or ``pd.Timedelta``, representing the sliding window
             size in terms of steps on the index (in case of a int or float) or the
             sliding window duration (in case of ``pd.Timedelta``).
        stride: Union[float, pd.TimeDelta]
            Either an int, float, or ``pd.Timedelta``, representing the sliding window
            size in terms of steps on the index (in case of a int or float) or the
            sliding window duration (in case of ``pd.Timedelta``).
        **kwargs:
            Additional keyword arguments, see the `StridedRolling` its documentation
            for more info.

        Returns
        -------
        StridedRolling
            The constructed StridedRolling instance.

        """
        data_dtype = AttributeParser.determine_type(data)
        args_dtype = AttributeParser.determine_type([window, stride])

        if data_dtype.value == args_dtype.value:
            return StridedRollingFactory._datatype_to_stroll[data_dtype](
                data, window, stride, **kwargs
            )
        elif data_dtype == DataType.TIME and args_dtype == DataType.SEQUENCE:
            # Note: this is very niche and thus requires advanced knowledge
            return TimeIndexSequenceStridedRolling(data, window, stride, **kwargs)
        elif data_dtype == DataType.SEQUENCE and args_dtype == DataType.TIME:
            raise ValueError("Cannot segment a sequence-series with a time window")
