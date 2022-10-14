"""AttributeParser class for determining the datatype of the given data."""

__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt"

import re
from enum import IntEnum
from typing import Any

import pandas as pd
import numpy as np

from tsflex.utils.time import parse_time_arg


class DataType(IntEnum):
    """Enum class for supported data types."""

    UNDEFINED = 0
    SEQUENCE = 1
    TIME = 2


class AttributeParser:
    """AttributeParser class for parsing the datatype of the given data."""

    _numeric_regexes = [re.compile(rf"{dtp}\d*") for dtp in ["float", "int", "uint"]]
    _datetime_regex = re.compile("datetime64+.")

    @staticmethod
    def determine_type(data: Any) -> DataType:
        """Determine the datatype of the given data."""
        if data is None:
            return DataType.UNDEFINED

        elif isinstance(data, (pd.Series, pd.DataFrame, np.ndarray)):
            if isinstance(data, np.ndarray):
                if not len(data):
                    return DataType.UNDEFINED
                dtype_str = str(data.dtype)
            else:
                dtype_str = str(data.index.dtype)
            if AttributeParser._datetime_regex.match(dtype_str) is not None:
                return DataType.TIME
            elif dtype_str == 'object':
                # we make the assumption that the fist element is the same type as the 
                # rest
                return AttributeParser.determine_type(data[0])
            elif any(r.match(dtype_str) for r in AttributeParser._numeric_regexes):
                return DataType.SEQUENCE

        elif isinstance(data, (int, float)):
            return DataType.SEQUENCE
        elif isinstance(data, pd.Timestamp):
            return DataType.TIME
        elif isinstance(data, (str, pd.Timedelta)):
            # parse_time_arg already raises an error when an invalid datatype is passed
            parse_time_arg(data)
            return DataType.TIME

        elif isinstance(data, list):
            dtype_list = [AttributeParser.determine_type(i) for i in data]
            if any(x != dtype_list[0] for x in dtype_list):
                raise ValueError(
                    f"Multiple dtypes for data {list(zip(dtype_list, data))}"
                )
            return dtype_list[0]

        raise ValueError(f"Unsupported data type {type(data)}")

    @staticmethod
    def check_expected_type(data: Any, expected: DataType) -> bool:
        """Check whether the given data has the expected datatype."""
        return AttributeParser.determine_type(data) == expected
