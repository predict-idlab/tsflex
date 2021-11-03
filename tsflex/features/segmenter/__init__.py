# -*- coding: utf-8 -*-
"""Series segmentation submodule."""

__author__ = "Jonas Van Der Donckt"

from ... import __pdoc__
from .strided_rolling import StridedRolling #TimeStridedRolling, SequenceStridedRolling
from .strided_rolling_factory import StridedRollingFactory

__all__ = [
    "StridedRolling",
    "StridedRollingFactory",
]
