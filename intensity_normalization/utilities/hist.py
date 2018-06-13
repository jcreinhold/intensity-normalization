#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smooth_hist


Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 13, 2018
"""

import logging

import numpy as np
from scipy.signal import argrelmax
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)


def smooth_hist(x, x_grid, bandwidths=None, cv=20):
    """
    use KDE to get smooth estimate of histogram

    Args:
        x (np.ndarray): flat array of data
        x_grid (np.ndarray): domain associated with data
        bandwidths:
        cv:

    Returns:
        smoothed (np.ndarray): smooth histogram (really a pdf)

    References:
       Jake VanderPlas, Kernel Density Estimation in Python,
       https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    """
    if bandwidths is None:
        bandwidths = np.linspace(0.1, 1.0, 30)
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': bandwidths},
                        cv=cv)  # 20-fold cross-validation
    grid.fit(x[:, None])
    kde = grid.best_estimator_
    log_pdf = kde.score_samples(x_grid[:, np.newaxis])
    import ipdb; ipdb.set_trace()
    return np.exp(log_pdf)


def get_largest_mode(data, num_pts=2000):
    """
    gets the last (reliable) peak in the histogram

    Args:
        bins (np.ndarray): bins of histogram (see np.histogram)
        counts (np.ndarray): counts of histogram (see np.histogram)

    Returns:
        largest_peak (int): index of the largest peak
    """
    grid = np.linspace(data.min(), data.max(), num_pts)
    sh = smooth_hist(data, grid)
    largest_peak = grid[np.argmax(sh)]
    return largest_peak


def get_last_mode(data, num_pts=2000, rare_prop=1/5, remove_tail=True):
    """
    gets the last (reliable) peak in the histogram

    Args:
        data (np.ndarray): bins of histogram (see np.histogram)
        num_pts (int): number of points in grid
        rare_prop (float): if remove_tail, use this proportion
        remove_tail (bool): remove rare portions of histogram
            (included to replicate the default behavior in the R version)

    Returns:
        last_peak (int): index of the last peak
    """
    if remove_tail:
        rare_thresh = np.percentile(data, 100*(1-rare_prop))
        which_rare = data >= rare_thresh
        data = data[which_rare != 1]
    grid = np.linspace(data.min(), data.max(), num_pts)
    sh = smooth_hist(data, grid)
    maxima = argrelmax(sh)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
    last_peak = grid[maxima[-1]]
    return last_peak


def get_first_mode(data, num_pts=2000, rare_prop=1/5, remove_tail=True):
    """
    gets the first (reliable) peak in the histogram

    Args:
        data (np.ndarray): bins of histogram (see np.histogram)
        num_pts (int): number of points in grid
        rare_prop (float): if remove_tail, use this proportion
        remove_tail (bool): remove rare portions of histogram
            (included to replicate the default behavior in the R version)

    Returns:
        first_peak (int): index of the first peak
    """
    if remove_tail:
        rare_thresh = np.percentile(data, 100*(1-rare_prop))
        which_rare = data >= rare_thresh
        data = data[which_rare != 1]
    grid = np.linspace(data.min(), data.max(), num_pts)
    sh = smooth_hist(data, grid)
    maxima = argrelmax(sh)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
    first_peak = grid[maxima[0]]
    return first_peak
