#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.kde

use kernel density estimation to find the peak of the histogram
associated with the WM and move this to peak to a (standard) value

Author: Blake Dewey (blake.dewey@jhu.edu),
        Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 24, 2018
"""

from __future__ import print_function, division

import logging

import nibabel as nib
import numpy as np
from scipy.signal import argrelextrema
import statsmodels.api as sm

from intensity_normalization.errors import NormalizationError

logger = logging.getLogger(__name__)


def kde_normalize(img, mask=None, contrast='T1', norm_value=1000):
    """
    use kernel density estimation to find the peak of the white
    matter in the histogram of a skull-stripped image. Normalize
    the WM of the non-skull-stripped image to norm_value

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        mask (nibabel.nifti1.Nifti1Image): brain mask of img
        contrast (str): contrast of img (T1,T1C,T2,PD,FL,FLC)
        norm_value (float): value at which to place WM peak

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): WM normalized img
    """
    if mask is not None:
        masked = img.get_data() * mask.get_data()
    else:
        masked = img.get_data()
    wm_peak = kde_wm_peak(masked, contrast)
    normalized = nib.Nifti1Image((img.get_data() / wm_peak) * norm_value,
                                 img.affine, img.header)
    return normalized


def kde_wm_peak(vol, contrast):
    """
    use kernel density estimate of histogram to find the
    the white matter peak in the histogram of a target image

    Args:
        vol (np.ndarray): target MR image data
        contrast (str): contrast of the target MR image

    Returns:
        x (float): WM peak in vol
    """
    temp = vol[np.nonzero(vol)]
    if contrast.upper() == 'T1C' or contrast.upper() == 'FLC':
        q = np.percentile(temp, 96.0)
    else:
        q = np.percentile(temp, 99.0)
    temp = temp[temp <= q]
    temp = np.asarray(temp, dtype=float).reshape(-1, 1)
    bw = float(q) / 80
    logger.info("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    X = 100.0 * kde.density
    Y = kde.support

    indx = argrelextrema(X, np.greater)
    indx = np.asarray(indx, dtype=int)
    H = X[indx]
    H = H[0]
    p = Y[indx]  # p is array of an array, so p=p[0] is necessary. I don't know why this happened!!!
    p = p[0]
    logger.info("%d peaks found." % (len(p)))
    if contrast.upper() in ("T1", "T1C"):
        x = p[-1]
        logger.info("Peak found at %.4f for %s" % (x, contrast))
    elif contrast.upper() in ("T2", "FL", "PD", "FLC"):
        x = np.amax(H)
        j = np.where(H == x)
        x = p[j]
        logger.info("Peak found at %.4f for %s" % (x, contrast))
    else:
        raise NormalizationError("Contrast must be one of T1,T1C,T2,PD,FL,FLC.")
    return x
