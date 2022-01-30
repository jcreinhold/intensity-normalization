"""Process the histograms of MR (brain) images
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

__all__ = [
    "get_first_tissue_mode",
    "get_largest_tissue_mode",
    "get_last_tissue_mode",
    "get_tissue_mode",
    "smooth_histogram",
]

import builtins
import typing

import numpy as np
import scipy.signal
import statsmodels.api as sm

import intensity_normalization as intnorm
import intensity_normalization.errors as intnorme
import intensity_normalization.typing as intnormt


def smooth_histogram(
    image: intnormt.Image, /
) -> typing.Tuple[intnormt.Image, intnormt.Image]:
    """Use kernel density estimate to get smooth histogram

    Args:
        image: array of image data (like an np.ndarray)

    Returns:
        grid: domain of the pdf
        pdf: kernel density estimate of the pdf of data
    """
    image_vec = np.asanyarray(image.flatten(), dtype=np.float64)
    bandwidth = image_vec.max() / 80
    kde = sm.nonparametric.KDEUnivariate(image_vec)
    kde.fit(kernel="gau", bw=bandwidth, gridsize=80, fft=True)
    pdf = 100.0 * kde.density
    grid = kde.support
    return grid, pdf


def get_largest_tissue_mode(image: intnormt.Image, /) -> builtins.float:
    """Mode of the largest tissue class

    Args:
        image: array of image data (like an np.ndarray)

    Returns:
        largest_tissue_mode: value of the largest tissue mode
    """
    grid, pdf = smooth_histogram(image)
    largest_tissue_mode: float = grid[np.argmax(pdf)]  # type: ignore[call-overload]
    return largest_tissue_mode


def get_last_tissue_mode(
    image: intnormt.Image,
    /,
    *,
    remove_tail: builtins.bool = True,
    tail_percentage: builtins.float = 96.0,
) -> builtins.float:
    """Mode of the highest-intensity tissue class

    Args:
        image: array of image data (like an np.ndarray)
        remove_tail: remove tail from histogram
        tail_percentage: if remove_tail, use the
            histogram below this percentage

    Returns:
        last_tissue_mode: mode of the highest-intensity tissue class
    """
    if not (0.0 < tail_percentage < 100.0):
        raise ValueError(f"tail_percentage must be in (0,100). Got {tail_percentage}.")
    if remove_tail:
        threshold: builtins.float = np.percentile(image, tail_percentage)  # type: ignore[call-overload]
        valid_mask: intnormt.Image = image <= threshold
        image = image[valid_mask]
    grid, pdf = smooth_histogram(image)
    maxima = scipy.signal.argrelmax(pdf)[0]
    last_tissue_mode: float = grid[maxima[-1]]
    return last_tissue_mode


def get_first_tissue_mode(
    image: intnormt.Image,
    /,
    *,
    remove_tail: builtins.bool = True,
    tail_percentage: builtins.float = 99.0,
) -> builtins.float:
    """Mode of the lowest-intensity tissue class

    Args:
        image: array of image data (like an np.ndarray)
        remove_tail: remove tail from histogram
        tail_percentage: if remove_tail, use the
            histogram below this percentage

    Returns:
        first_tissue_mode: mode of the lowest-intensity tissue class
    """
    if not (0.0 < tail_percentage < 100.0):
        raise ValueError(f"tail_percentage must be in (0,100). Got {tail_percentage}.")
    if remove_tail:
        threshold: builtins.float = np.percentile(image, tail_percentage)  # type: ignore[call-overload]
        valid_mask: intnormt.Image = image <= threshold
        image = image[valid_mask]
    grid, pdf = smooth_histogram(image)
    maxima = scipy.signal.argrelmax(pdf)[0]
    first_tissue_mode: float = grid[maxima[0]]
    return first_tissue_mode


def get_tissue_mode(
    image: intnormt.Image, /, *, modality: intnormt.Modalities
) -> builtins.float:
    """Find the appropriate tissue mode given a modality"""
    modality_ = modality.value
    if modality_ in intnorm.PEAK["last"]:
        mode = get_last_tissue_mode(image)
    elif modality_ in intnorm.PEAK["largest"]:
        mode = get_largest_tissue_mode(image)
    elif modality_ in intnorm.PEAK["first"]:
        mode = get_first_tissue_mode(image)
    else:
        modalities = ", ".join(intnorm.VALID_PEAKS)
        msg = f"Modality {modality} not valid. Needs to be one of {modalities}."
        raise intnorme.NormalizationError(msg)
    return mode
