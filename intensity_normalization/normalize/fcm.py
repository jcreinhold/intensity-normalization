#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.fcm

use fuzzy c-means to find a mask for a specified tissue type
given a T1w image and it's brain mask. Create a tissue mask
from that T1w image's FCM tissue mask. Then we can use that
tissue mask as input to the func again, where the tissue mask is
used to find an approximate mean of the tissue intensity in
another target contrast, and move it to some standard value.

Author: Blake Dewey (blake.dewey@jhu.edu),
        Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 24, 2018
"""

from __future__ import print_function, division

import logging

import nibabel as nib

from intensity_normalization.utilities import mask

logger = logging.getLogger(__name__)


def fcm_normalize(img, tissue_mask, norm_value=1):
    """
    Use FCM generated mask to normalize some specified tissue of a target image

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        tissue_mask (nibabel.nifti1.Nifti1Image): tissue mask for img
        norm_value (float): value at which to place the tissue mean

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with specified tissue mean at norm_value
    """

    img_data = img.get_data()
    tissue_mask_data = tissue_mask.get_data()
    tissue_mean = img_data[tissue_mask_data > 0].mean()
    normalized = nib.Nifti1Image((img_data / tissue_mean) * norm_value,
                                 img.affine, img.header)
    return normalized


def find_tissue_mask(img, brain_mask, threshold=0.8, tissue_type='wm'):
    """
    find tissue mask using FCM with a membership threshold

    Args:
        img (nibabel.nifti1.Nifti1Image): target img
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
        threshold (float): membership threshold
        tissue_type (str): find the mask of this tissue type (wm, gm, or csf)

    Returns:
        tissue_mask_nifti (nibabel.nifti1.Nifti1Image): tissue mask for img
    """
    tissue_to_int = {'csf': 0, 'gm': 1, 'wm': 2}
    t1_mem = mask.fcm_class_mask(img, brain_mask)
    tissue_mask = t1_mem[..., tissue_to_int[tissue_type]] > threshold
    tissue_mask_nifti = nib.Nifti1Image(tissue_mask, img.affine, img.header)
    return tissue_mask_nifti
