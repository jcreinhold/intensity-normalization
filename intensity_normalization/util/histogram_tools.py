# -*- coding: utf-8 -*-
"""
intensity_normalization.util.histogram_tools

Process the histograms of MR (brain) images

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "get_first_tissue_mode",
    "get_largest_tissue_mode",
    "get_last_tissue_mode",
    "get_tissue_mode",
    "smooth_histogram",
]

from typing import Tuple

import numpy as np
import statsmodels.api as sm
from scipy.signal import argrelmax

from intensity_normalization import PEAK, VALID_PEAKS
from intensity_normalization.errors import NormalizationError
from intensity_normalization.type import Array


def smooth_histogram(data: Array) -> Tuple[Array, Array]:
    """Use kernel density estimate to get smooth histogram

    Args:
        data: array of image data

    Returns:
        grid: domain of the pdf
        pdf: kernel density estimate of the pdf of data
    """
    data_vec = data.flatten().astype(np.float64)
    bandwidth = data_vec.max() / 80  # noqa
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit(kernel="gau", bw=bandwidth, gridsize=80, fft=True)
    pdf = 100.0 * kde.density
    grid = kde.support
    return grid, pdf


def get_largest_tissue_mode(data: Array) -> float:
    """Mode of the largest tissue class

    Args:
        data: image data

    Returns:
        largest_tissue_mode (float): intensity of the mode
    """
    grid, pdf = smooth_histogram(data)
    largest_tissue_mode: float = grid[np.argmax(pdf)]
    return largest_tissue_mode


def get_last_tissue_mode(
    data: Array,
    remove_tail: bool = True,
    tail_percentage: float = 96.0,
) -> float:
    """Mode of the highest-intensity tissue class

    Args:
        data: image data
        remove_tail: remove tail from histogram
        tail_percentage: if remove_tail, use the
            histogram below this percentage

    Returns:
        last_tissue_mode: mode of the highest-intensity tissue class
    """
    if remove_tail:
        threshold = np.percentile(data, tail_percentage)
        valid_mask = data <= threshold
        data = data[valid_mask]
    grid, pdf = smooth_histogram(data)
    maxima = argrelmax(pdf)[0]
    last_tissue_mode: float = grid[maxima[-1]]
    return last_tissue_mode


def get_first_tissue_mode(
    data: Array,
    remove_tail: bool = True,
    tail_percentage: float = 99.0,
) -> float:
    """Mode of the lowest-intensity tissue class

    Args:
        data: image data
        remove_tail: remove tail from histogram
        tail_percentage: if remove_tail, use the
            histogram below this percentage

    Returns:
        first_tissue_mode: mode of the lowest-intensity tissue class
    """
    if remove_tail:
        threshold = np.percentile(data, tail_percentage)
        valid_mask = data <= threshold
        data = data[valid_mask]
    grid, pdf = smooth_histogram(data)
    maxima = argrelmax(pdf)[0]
    first_tissue_mode: float = grid[maxima[0]]
    return first_tissue_mode


def get_tissue_mode(data: Array, modality: str) -> float:
    """Find the appropriate tissue mode given a modality"""
    modality_ = modality.lower()
    if modality_ in PEAK["last"]:
        mode = get_last_tissue_mode(data)
    elif modality_ in PEAK["largest"]:
        mode = get_largest_tissue_mode(data)
    elif modality_ in PEAK["first"]:
        mode = get_first_tissue_mode(data)
    else:
        modalities = ", ".join(VALID_PEAKS)
        msg = f"Modality {modality} not valid. Needs to be one of {modalities}."
        raise NormalizationError(msg)
    return mode
