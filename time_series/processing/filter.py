# -*- coding: utf-8 -*-
"""
    ****************
    filter.py
    ****************
    
    Created at 17/06/20        
"""
__author__ = 'Jonas Van Der Donckt'

import numpy as np
import pandas as pd
import scipy.signal as signal


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data).astype(np.float32)


def notch_filter(data, f_notch, fs, q=30):
    b, a = signal.iirnotch(w0=f_notch, Q=q, fs=fs)
    return signal.lfilter(b, a, data).astype(np.float32)


def low_pass_filter(sig: pd.Series, order: int = 5, f_cutoff: int = 1, fs: int = None) -> np.ndarray:
    if fs is None:  # determine the sample frequency
        fs = 1 / pd.Timedelta(pd.infer_freq(sig.index)).total_seconds()
    b, a = signal.butter(N=order, Wn=f_cutoff / (0.5 * fs), btype='lowpass', output='ba', fs=fs)
    # the filtered output has the same shape as sig.values
    return signal.filtfilt(b=b, a=a, x=sig.values).astype(np.float32)

# def high_pass_filter(sig: pd.Series, order=)
