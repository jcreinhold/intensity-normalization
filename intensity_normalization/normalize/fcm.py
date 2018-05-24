#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.fcm

use fuzzy c-means to find a mask for the white matter
given a T1w image and it's brain mask. Create a WM mask
from that T1w image's FCM WM mask. Then we can use that
WM mask as input to the func again, where the WM mask is
used to find an approximate mean of the WM intensity in
another target contrast, move it to some standard value.

Author: Blake Dewey (blake.dewey@jhu.edu)
        Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 24, 2018
"""

from __future__ import print_function, division

import logging

import nibabel as nib

from intensity_normalization.utilities import mask

logger = logging.getLogger(__name__)


def fcm_normalize(img, wm_mask, norm_value=1000):
    """
    Use FCM generated mask to normalize the WM of a target image

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        wm_mask (nibabel.nifti1.Nifti1Image): white matter mask for img
        norm_value (float): value at which to place the WM mean

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    img_data = img.get_data()
    wm_mask_data = wm_mask.get_data()
    wm_mean = img_data[wm_mask_data == 1].mean()
    normalized = nib.Nifti1Image((img_data / wm_mean) * norm_value,
                                 img.affine, img.header)
    return normalized


def find_wm_mask(img, brain_mask, threshold=0.8):
    """
    find WM mask using FCM with a membership threshold

    Args:
        img (nibabel.nifti1.Nifti1Image): target img
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
        threshold (float): membership threshold

    Returns:
        wm_mask (nibabel.nifti1.Nifti1Image): white matter mask for img
    """
    t1_mem = mask.fcm_class_mask(img, brain_mask)
    wm_mask = t1_mem[..., 2] > threshold
    wm_mask_nifti = nib.Nifti1Image(wm_mask, img.affine, img.header)
    return wm_mask_nifti
