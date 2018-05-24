#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.ravel

Use RAVEL [1] to intensity normalize a population of MR images

Note that this package requires RAVEL (and its dependencies)
to be installed in R

References:
   ﻿[1] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu,
        and R. T. Shinohara, “Removing inter-subject technical variability
        in magnetic resonance imaging studies,” Neuroimage, vol. 132,
        pp. 198–212, 2016.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 27, 2018
"""

from functools import reduce
from glob import glob
import logging
from operator import add
import os

import numpy as np
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from rpy2.rinterface import NULL

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import io
from intensity_normalization.utilities.mask import csf_mask

ravel = importr('RAVEL')

logger = logging.getLogger(__name__)


def ravel_normalize(img_dir, template_mask, csf_mask, contrast, output_dir=None, write_to_disk=True, **kwargs):
    """
    Use RAVEL [1] to normalize the intensities of a set of MR images to eliminate
    unwanted technical variation in images (but, hopefully, preserve biological variation)

    Args:
        img_dir (str): directory containing MR images to be normalized
        template_mask (str): brain mask for template image
        csf_mask (str): path to csf mask for data in data_dir
        contrast (str): contrast of MR images to be normalized (T1, T2, or FLAIR)
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        write_to_disk (bool): write the normalized data to disk or nah
        kwargs: ravel keyword arguments not included here

    Returns:
        normalized (np.ndarray): set of normalized images from data_dir

    References:
        [1] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu,
            and R. T. Shinohara, “Removing inter-subject technical variability
            in magnetic resonance imaging studies,” Neuroimage, vol. 132,
            pp. 198–212, 2016.
    """
    data = sorted(glob(os.path.join(img_dir, '*.nii*')))
    input_files = StrVector(data)
    if output_dir is None:
        output_files = NULL
    else:
        out_fns = []
        for fn in data:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + ext))
        output_files = StrVector(out_fns)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    verbose = True if logger.getEffectiveLevel() == logging.getLevelName('DEBUG') else False

    normalizedR = ravel.normalizeRAVEL(input_files, control_mask=csf_mask,
                                       output_files=output_files, brain_mask=template_mask,
                                       WhiteStripe_Type=contrast, writeToDisk=write_to_disk,
                                       returnMatrix=True, verbose=verbose, **kwargs)
    normalized = np.array(normalizedR)
    return normalized


def csf_mask_intersection(img_dir, mask_dir=None, prob=0.9):
    """
    use all nifti T1w images in data_dir to create csf mask in common areas

    Args:
        img_dir (str): directory containing MR images to be normalized
        mask_dir (str): if images are not skull-stripped, then provide brain mask
        prob (float): given all data, proportion of data labeled as csf to be
            used for intersection

    Returns:
        intersection (np.ndarray): binary mask of common csf areas for all provided imgs
    """
    if not (0 <= prob <= 1):
        raise NormalizationError('prob must be between 0 and 1. {} given.'.format(prob))
    data = sorted(glob(os.path.join(img_dir, '*.nii*')))
    if mask_dir is None:
        masks = [None] * len(data)
    else:
        masks = sorted(glob(os.path.join(mask_dir, '*.nii*')))
    logger.info('creating csf masks for all images')
    csf = [csf_mask(io.open_nii(img), brain_mask=io.open_nii(mask)) for img, mask in zip(data, masks)]
    csf_sum = reduce(add, csf)  # need to use reduce instead of sum b/c data structure
    intersection = np.zeros(csf_sum.shape)
    intersection[csf_sum > np.floor(len(data) * prob)] = 1
    return intersection
