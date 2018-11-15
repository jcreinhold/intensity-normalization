#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.zscore

normalize an image by simply subtracting the mean
and dividing by the standard deviation of the whole brain

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 30, 2018
"""

from __future__ import print_function, division

import logging

import nibabel as nib

logger = logging.getLogger(__name__)


def zscore_normalize(img, mask=None):
    """
    normalize a target image by subtracting the mean of the whole brain
    and dividing by the standard deviation

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        mask (nibabel.nifti1.Nifti1Image): brain mask for img

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    img_data = img.get_data()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_data()
    elif mask == 'nomask':
        mask_data = img_data == img_data
    else:
        mask_data = img_data > img_data.mean()
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    return normalized
