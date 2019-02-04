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

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import hist

logger = logging.getLogger(__name__)


def kde_normalize(img, mask=None, contrast='t1', norm_value=1):
    """
    use kernel density estimation to find the peak of the white
    matter in the histogram of a skull-stripped image. Normalize
    the WM of the non-skull-stripped image to norm_value

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        mask (nibabel.nifti1.Nifti1Image): brain mask of img
        contrast (str): contrast of img (T1,T2,FA,MD)
        norm_value (float): value at which to place WM peak

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): WM normalized img
    """
    if mask is not None:
        voi = img.get_data()[mask.get_data() == 1].flatten()
    else:
        voi = img.get_data()[img.get_data() > img.get_data().mean()].flatten()
    if contrast.lower() in ['t1', 'flair', 'last']:
        wm_peak = hist.get_last_mode(voi)
    elif contrast.lower() in ['t2', 'largest']:
        wm_peak = hist.get_largest_mode(voi)
    elif contrast.lower() in ['md', 'first']:
        wm_peak = hist.get_first_mode(voi)
    else:
        raise NormalizationError('Contrast {} not valid, needs to be `t1`,`t2`,`flair`,`md`,`first`,`largest`,`last`'.format(contrast))
    normalized = nib.Nifti1Image((img.get_data() / wm_peak) * norm_value,
                                 img.affine, img.header)
    return normalized
