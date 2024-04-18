"""Object-oriented utilities."""

__author__ = "Jonas Van Der Donckt"

from typing import Any


class FrozenClass(object):
    """Superclass which allows subclasses to freeze at any time."""

    __is_frozen = False

    def __setattr__(self, key: Any, value: Any) -> None:
        if self.__is_frozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self) -> None:
        self.__is_frozen = True
