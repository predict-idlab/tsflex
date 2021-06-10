# -*- coding: utf-8 -*- # TODO: zetten we dat nu overal of niet?
"""Object-oriented utilities."""

__author__ = 'Jonas Van Der Donckt'


class FrozenClass(object):
    """Superclass which allows subclasses to freeze at any time."""

    __is_frozen = False

    def __setattr__(self, key, value):
        if self.__is_frozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__is_frozen = True
