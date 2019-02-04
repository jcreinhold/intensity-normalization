#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.gmm

fit three gaussians to the histogram of
skull-stripped image and normalize the WM mean
to some standard value

Author: Blake Dewey (blake.dewey@jhu.edu),
        Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 24, 2018
"""

from __future__ import print_function, division

import logging

import nibabel as nib
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    from sklearn.mixture import GMM as GaussianMixture

from intensity_normalization.utilities.mask import gmm_class_mask

logger = logging.getLogger(__name__)


def gmm_normalize(img, brain_mask=None, norm_value=1, contrast='t1', bg_mask=None, wm_peak=None):
    """
    normalize the white matter of an image using a GMM to find the tissue classes

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
        norm_value (float): value at which to place the WM mean
        contrast (str): MR contrast type for img
        bg_mask (nibabel.nifti1.Nifti1Image): if provided, use to zero bkgd
        wm_peak (float): previously calculated WM peak

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): gmm wm peak normalized image
    """

    if wm_peak is None:
        wm_peak = gmm_class_mask(img, brain_mask=brain_mask, contrast=contrast)

    img_data = img.get_data()
    logger.info('Normalizing Data...')
    norm_data = (img_data/wm_peak)*norm_value
    norm_data[norm_data < 0.1] = 0.0
    
    if bg_mask is not None:
        logger.info('Applying background mask...')
        masked_image = norm_data * bg_mask.get_data()
    else:
        masked_image = norm_data

    normalized = nib.Nifti1Image(masked_image, img.affine, img.header)
    return normalized
