#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.quality

examine the quality/consistency of the intensity normalization

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 04, 2018
"""

from __future__ import print_function, division

import logging
import warnings

import nibabel as nib
import numpy as np

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import io

logger = logging.getLogger(__name__)


def jsd(p, q):
    """
    Jensen-Shannon Divergence for two histograms

    Args:
        p (np.ndarray): histogram 1
        q (np.ndarray): histogram 2

    Returns:
        D_js (float): Jensen-Shannon divergence of p and q
    """
    m = 1/2 * (p + q)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        D_js = 1/2 * np.sum(p * np.log10(p / m)) + 1/2 * np.sum(q * np.log10(q / m))
    return D_js


def pairwise_jsd(img_dir, mask_dir, nbins=200):
    """
    Calculate the Jensen-Shannon Divergence for all pairs of images in the image directory

    Args:
        img_dir (str): path to directory of images
        mask_dir (str): path to directory of masks
        nbins (int): number of bins to use in the histograms

    Returns:
        pairwise_jsd (np.ndarray): array of pairwise Jensen-Shannon divergence
    """
    eps = np.finfo(np.float32).eps

    img_fns = io.glob_nii(img_dir)
    mask_fns = io.glob_nii(mask_dir)

    if len(img_fns) != len(mask_fns):
        raise NormalizationError(f'Number of images ({len(img_fns)}) must be equal to the number of masks ({len(mask_fns)}).')

    min_intensities, max_intensities = [], []
    for img_fn, mask_fn in zip(img_fns, mask_fns):
        data = nib.load(img_fn).get_data()[nib.load(mask_fn).get_data() == 1]
        min_intensities.append(np.min(data))
        max_intensities.append(np.max(data))
    intensity_range = (min(min_intensities), max(max_intensities))

    hists = []
    for img_fn, mask_fn in zip(img_fns, mask_fns):
        data = nib.load(img_fn).get_data()[nib.load(mask_fn).get_data() == 1]
        hist, _ = np.histogram(data.flatten(), nbins, range=intensity_range, density=True)
        hists.append(hist + eps)

    pairwise_jsd = []
    for i in range(len(hists)):
        for j in range(i + 1, len(hists)):
            pairwise_jsd.append(jsd(hists[i], hists[j]))

    return np.array(pairwise_jsd)
