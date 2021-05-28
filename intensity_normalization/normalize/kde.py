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

import logging

import nibabel as nib

from intensity_normalization.utilities import hist

logger = logging.getLogger(__name__)


def kde_normalize(image, mask=None, modality="t1", norm_value=1):
    """
    use kernel density estimation to find the peak of the white
    matter in the histogram of a skull-stripped image. Normalize
    the WM of the non-skull-stripped image to norm_value

    Args:
        image (nibabel.nifti1.Nifti1Image): target MR image
        mask (nibabel.nifti1.Nifti1Image): brain mask of img
        modality (str): modality of the MR image (e.g., t1)
        norm_value (float): value at which to place WM mode

    Returns:
        normalized (nibabel.nifti1.Nifti1Image):
            WM mode normalized image
    """
    data = image.get_fdata()
    if mask is not None:
        mask_data = mask.get_fdata() > 0.0
        voi = data[mask_data].flatten()  # noqa
    else:
        voi = data[data > data.mean()].flatten()  # noqa
    peak = hist.get_tissue_mode(voi, modality)
    normalized_data = (image.get_fdata() / peak) * norm_value  # noqa
    normalized = nib.Nifti1Image(normalized_data, image.affine, image.header)
    return normalized
