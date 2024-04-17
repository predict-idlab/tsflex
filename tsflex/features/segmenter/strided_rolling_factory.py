"""
Factory class for creating the proper StridedRolling instances.

.. TODO::
    Also create a (SegmenterFactory) which the StridedRollingFactory implements

"""

__author__ = "Jonas Van Der Donckt"

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
            When incompatible data & window-stride data types are passed (e.g. time
            window-stride args on sequence data-index).

        Returns
        -------
        StridedRolling
            The constructed StridedRolling instance.

        """
        data_dtype = AttributeParser.determine_type(data)
        if strides is None:
            args_dtype = AttributeParser.determine_type(window)
        else:
            args_dtype = AttributeParser.determine_type([window] + strides)

        if window is None or data_dtype.value == args_dtype.value:
            return StridedRollingFactory._datatype_to_stroll[data_dtype](
                data, window, strides, **kwargs
            )
        elif data_dtype == DataType.TIME and args_dtype == DataType.SEQUENCE:
            # Note: this is very niche and thus requires advanced knowledge
            return TimeIndexSampleStridedRolling(data, window, strides, **kwargs)
        elif data_dtype == DataType.SEQUENCE and args_dtype == DataType.TIME:
            raise ValueError("Cannot segment a sequence-series with a time window")

        # This should never happen
        raise ValueError(
            f"Cannot segment data of type {data_dtype} with window-stride of type {args_dtype}"
        )
