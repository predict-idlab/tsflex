# -*- coding: utf-8 -*-
"""
Factory class for creating the proper StridedRolling instances.

.. TODO::
    Maybe also create a (SegmenterFactory) which the StridedRollingFactory implements

"""
__author__ = "Jonas Van Der Donckt"

import re

from .strided_rolling import StridedRolling, TimeStridedRolling, SequenceStridedRolling
from ...utils.data import to_series_list


class StridedRollingFactory:
    @staticmethod
    def get_segmenter(data, **kwargs) -> StridedRolling:
        """Get the appropriate StridedRolling instance for the passed data.

        The returned instance will be determined by the data its index type

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
            The data to segment.
        **kwargs:
            Additional keyword arguments, see the `StridedRolling` its documentation
            for more info.


        Returns
        -------
        StridedRolling
            The constructed StridedRolling instance.

        """
        numeric_regexes = [re.compile(rf"{dtp}\d*") for dtp in ["float", "int", "uint"]]
        datetime_regex = re.compile("datetime64+.")

        dtypes = [str(s.index.dtype) for s in to_series_list(data)]

        # Verify whether all series indexes are of the same type
        if all([any([p.match(dtp) for p in numeric_regexes]) for dtp in dtypes]):
            return SequenceStridedRolling(data, **kwargs)
        elif all([datetime_regex.match(dtp) for dtp in dtypes]):
            return TimeStridedRolling(data, **kwargs)
        else:
            raise ValueError(f"invalid data-index types {dtypes}")
