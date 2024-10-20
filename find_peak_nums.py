# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:41:45 2024

@author: prism
"""

from scipy.signal import find_peaks

def find_peak_nums(results):
    peaks, _ = find_peaks(results[:, 1], prominence=10/5000)
    return len(peaks)
