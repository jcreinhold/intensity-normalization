#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.hist

holds routines to process histograms of MR neuro images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 13, 2018
"""

import numpy as np
from scipy.signal import argrelmax
import statsmodels.api as sm


def smooth_hist(data):
    """
    use KDE to get smooth estimate of histogram

    Args:
        data (np.ndarray): array of image data

    Returns:
        grid (np.ndarray): domain of the pdf
        pdf (np.ndarray): kernel density estimate of the pdf of data
    """
    data = data.flatten().astype(np.float64)
    bw = data.max() / 80

    kde = sm.nonparametric.KDEUnivariate(data)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    pdf = 100.0 * kde.density
    grid = kde.support

    return grid, pdf


def get_largest_mode(data):
    """
    gets the last (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data

    Returns:
        largest_peak (int): index of the largest peak
    """
    grid, pdf = smooth_hist(data)
    largest_peak = grid[np.argmax(pdf)]
    return largest_peak


def get_last_mode(data, rare_prop=96, remove_tail=True):
    """
    gets the last (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data
        rare_prop (float): if remove_tail, use the proportion of hist above
        remove_tail (bool): remove rare portions of histogram
            (included to replicate the default behavior in the R version)

    Returns:
        last_peak (int): index of the last peak
    """
    if remove_tail:
        rare_thresh = np.percentile(data, rare_prop)
        which_rare = data >= rare_thresh
        data = data[which_rare != 1]
    grid, pdf = smooth_hist(data)
    maxima = argrelmax(pdf)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
    last_peak = grid[maxima[-1]]
    return last_peak


def get_first_mode(data, rare_prop=99, remove_tail=True):
    """
    gets the first (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data
        rare_prop (float): if remove_tail, use the proportion of hist above
        remove_tail (bool): remove rare portions of histogram
            (included to replicate the default behavior in the R version)

    Returns:
        first_peak (int): index of the first peak
    """
    if remove_tail:
        rare_thresh = np.percentile(data, rare_prop)
        which_rare = data >= rare_thresh
        data = data[which_rare != 1]
    grid, pdf = smooth_hist(data)
    maxima = argrelmax(pdf)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
    first_peak = grid[maxima[0]]
    return first_peak
