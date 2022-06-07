"""<i><b>flex</b>ible <b>t</b>ime-<b>s</b>eries operations</i>

.. include:: ../docs/pdoc_include/root_documentation.md


.. include:: ../docs/pdoc_include/tsflex_matrix.md

"""

__docformat__ = 'numpy'
__author__ = "Jonas Van Der Donckt, Jeroen Van Der Donckt, Emiel Deprost"
__version__ = '0.2.3.7'
__pdoc__ = {
    # do not show tue utils module
    'tsflex.utils': False,
    # show the seriesprocessor & funcwrapper their call method
    'SeriesProcessor.__call__': True,
    'FuncWrapper.__call__': True,
}

__all__ = ["__version__", "__pdoc__"]
