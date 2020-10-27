# -*- coding: utf-8 -*-
"""
    ************
    feature_extraction.py
    ************

    
"""
__author__ = 'Jonas Van Der Donckt'

from typing import List, Dict, Tuple

import pandas as pd

from .feature import NumpyFeatureCalculation
from ..strided_rolling import StridedRolling


class NumpyFeatureCalculationRegistry:
    """Returns a DataFrame for each different win_size_s, stride_combination"""

    def __init__(self, features: List[NumpyFeatureCalculation] = None):
        self.feature_registry = features if features is not None else []

    def append(self, feature: NumpyFeatureCalculation):
        """Adds a feature to the registry"""
        self.feature_registry.append(feature)

    def _create_win_stride_dict(self) -> Dict[Tuple[int, int], List[NumpyFeatureCalculation]]:
        """Constructs a dict in which features with the same window and stride are clustered together """
        win_stride_dict = {}
        for feat in self.feature_registry:
            w_s = feat.get_win_stride()
            if w_s in win_stride_dict.keys():
                win_stride_dict[w_s].append(feat)
            else:
                win_stride_dict[w_s] = [feat]
        return win_stride_dict

    def calculate_features(self, time_series_df: pd.DataFrame) -> Dict[Tuple[int, int], pd.DataFrame]:
        """Calculates the features for a time_series indexed DataFrame

        :param time_series_df: The time indexed series / DataFrame (containing one column)
            for which the features are calculated
        :return: A dict with key a tuple of (window_size, stride) respectively and the feature Dat
        """
        # Order the features by win_stride and iterate over the various win_stride
        win_stride_dict = self._create_win_stride_dict()
        feat_dict = {}
        for (win_size, stride), features in win_stride_dict.items():
            df_feat = StridedRolling(time_series_df, window=win_size, stride=stride).apply_funcs(features)
            feat_dict[(win_size, stride)] = df_feat
        return feat_dict

    def __repr__(self) -> str:
        repr_str = ""
        for (win_size, stride), features in self._create_win_stride_dict().items():
            repr_str += f"\twin: {win_size}, stride: {stride}: ["
            repr_str += ''.join(['\n\t\t' + str(f) + ',  ' for f in features])
            repr_str += '\n\t]\n'
        return repr_str

    def __str__(self):
        return self.__repr__()


class NumpyFeatureCalculationPipeline:
    """Wrapper around the NumpyFeatureCalculationRegistry class

    Supports functionality to calculate features for multiple signals
    """

    def __init__(self, df_feature_wrappers: List[Tuple[str, NumpyFeatureCalculationRegistry]] = None):
        self.sig_feature_registry = df_feature_wrappers if df_feature_wrappers is not None else []

    def append(self, sig_str: str, df_featurewrapper: NumpyFeatureCalculationRegistry) -> None:
        self.sig_feature_registry.append((sig_str, df_featurewrapper))

    def __call__(self, df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Cal(l)culates the features"""
        dfs, win_strides = [], []
        for sig_str, df_feature_wrapper in self.sig_feature_registry:
            win_stride_df_feat_dict = df_feature_wrapper.calculate_features(df_dict[sig_str])
            win_strides += list(win_stride_df_feat_dict.keys())
            dfs += list(win_stride_df_feat_dict.values())

        # sort them on descending size & merge into participant DataFrame
        dfs.sort(key=lambda x: x.shape[0], reverse=True)
        df_tot = None
        df: pd.DataFrame
        for df, (win, stride) in zip(dfs, win_strides):
            df = df.add_suffix(suffix=f'__w={win}_s={stride}')
            df_tot = df if df_tot is None else pd.merge_asof(left=df_tot, right=df, left_index=True, right_index=True,
                                                             direction='nearest')
        return df_tot

    def __repr__(self):
        repr_str = ""
        for sig, feat_registry in self.sig_feature_registry:
            repr_str += f"{sig}: (\n" + str(feat_registry) + '\n)\n'
        return repr_str

    def __str__(self):
        return self.__repr__()
