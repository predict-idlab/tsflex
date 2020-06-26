# -*- coding: utf-8 -*-
"""
    *************
    processor.py
    *************

    
"""
__author__ = 'Jonas Van Der Donckt'

# Rubber ducking with myself:
#   mhm we hebben iets van logica nodig om bij te houden welke signalen (dict keys) we nodig hebben voor
#   een signaal te processen en welke signaal (output) dict we processen
#   --> ik zou dan voorstellen om alle processing shizzles als methodes te schrijven en deze dan te injecteren
# TODO: this code can be written cleaner (more pipeline alike, maybe even a wrapper around func ...


from itertools import chain
from typing import Dict, List, Union

import pandas as pd


class DictProcessor:
    def __init__(self, required_signals: List[str], func, **kwargs):
        """
        :param required_signals: The signals required to perform the operations
        :param func: The feature calculation func, takes a dict with keys the signal names and the corresponding
                (time indexed) DataFrame as input, & outputs a DataFrame dict
        :param kwargs: Additional kwargs for the func
        """
        self.required_signals = required_signals
        self.func = func
        self.kwargs = kwargs

    def __call__(self, series_dict: Dict[str, Union[pd.DataFrame, pd.Series]]) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Cal(l)culates the processed signal
        :param series_dict: The multimodal DataFrame dict
        """
        # Only selecting the signals that are needed for this processing step
        requested_dict = {}
        try:
            for sig in self.required_signals:
                requested_dict[sig] = series_dict[sig]
        except KeyError as key:
            # Re raise error as we can't continue
            raise KeyError("Key %s is not present in the input dict" % (key))

        return self.func(requested_dict, **self.kwargs) if self.kwargs is not None else self.func(series_dict)

    def __repr__(self):
        return self.func.__name__ + (' ' + str(self.kwargs)) if self.kwargs is not None else ''

    def __str__(self):
        return self.__repr__()


class DictProcessorWrapper:
    """Processes the data_dict signals in a sequential manner, determined by the processing code"""

    def __init__(self, processors: List[DictProcessor] = None):
        processors = [] if processors is None else processors
        self.processing_registry: List[DictProcessor] = processors

    def get_all_required_signals(self) -> List[str]:
        """Returns a  list of all required signal keys for this DictProcessorWrapper"""
        return list(set(chain.from_iterable([pr.required_signals for pr in self.processing_registry])))

    def append(self, processor: DictProcessor) -> None:
        """Append a processor to the registry"""
        self.processing_registry.append(processor)

    def process(self, series_dict: Dict[str, Union[pd.DataFrame, pd.Series]]) -> Dict[str, pd.DataFrame]:
        """Applies all the processing steps on the series_dict and returns the resulting dict."""
        series_dict = series_dict.copy()
        for processor in self.processing_registry:
            series_dict.update(processor(series_dict))
        return series_dict

    def __call__(self, series_dict: Dict[str, Union[pd.DataFrame, pd.Series]]) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        return self.process(series_dict)

    def __repr__(self):
        return "[\n" + ''.join([f'\t{str(pr)}\n' for pr in self.processing_registry]) + "]"

    def __str__(self):
        return self.__repr__()
