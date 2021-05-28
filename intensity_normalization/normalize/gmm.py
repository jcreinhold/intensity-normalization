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

import logging

import nibabel as nib

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    from sklearn.mixture import GMM as GaussianMixture

from intensity_normalization.utilities.mask import gmm_class_mask

logger = logging.getLogger(__name__)


def gmm_normalize(
    image, brain_mask=None, norm_value=1, modality="t1", bg_mask=None, wm_mean=None
):
    """
    normalize the white matter of an image using a GMM to find the tissue classes

    Args:
        image (nibabel.nifti1.Nifti1Image): MR image to be normalized
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for image
        norm_value (float): value at which to place the WM mean
        modality (str): MR contrast type for image
        bg_mask (nibabel.nifti1.Nifti1Image): if provided, use to zero background
        wm_mean (float): previously calculated WM mean

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): gmm WM mean normalized image
    """

    if wm_mean is None:
        wm_mean = gmm_class_mask(image, brain_mask=brain_mask, modality=modality)

    data = image.get_fdata()
    normalized_data = (data / wm_mean) * norm_value

    if bg_mask is not None:
        masked_image = normalized_data * bg_mask.get_fdata()
    else:
        masked_image = normalized_data

    normalized = nib.Nifti1Image(masked_image, image.affine, image.header)
    return normalized
