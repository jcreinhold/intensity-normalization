#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.zscore

normalize an image by simply subtracting the mean
and dividing by the standard deviation of the whole brain

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 30, 2018
"""

import logging

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def zscore_normalize(image, mask=None):
    """
    normalize a target image by subtracting the mean and dividing
    by the standard deviation of the foreground

    Args:
        image (nibabel.nifti1.Nifti1Image): MR image to normalize
        mask (nibabel.nifti1.Nifti1Image): foreground mask for image

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): normalized image
    """

    data = image.get_fdata()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_fdata()
    elif mask == "nomask":
        mask_data = np.ones_like(data)
    else:
        mask_data = data > data.mean()  # noqa
    logical_mask = mask_data > 0.0  # force the mask to be logical type
    data_inside_mask = data[logical_mask]  # noqa
    mean = data_inside_mask.mean()
    std = data_inside_mask.std()
    normalized_data = (data - mean) / std  # noqa
    normalized = nib.Nifti1Image(normalized_data, image.affine, image.header)
    return normalized
