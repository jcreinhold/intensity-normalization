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

import argparse
from functools import reduce
from glob import glob
import logging
from operator import add
import os
import sys

import nibabel as nib
import numpy as np
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from rpy2.rinterface import NULL

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import io
from intensity_normalization.utilities.mask import csf_mask

ravel = importr('RAVEL')

logger = logging.getLogger()


def ravel_normalize(data_dir, csf_mask, contrast, mask_dir=None, output_dir=None, write_to_disk=True):
    """
    Use RAVEL [1] to normalize the intensities of a set of MR images to eliminate
    unwanted technical variation in images (but, hopefully, preserve biological variation)

    Args:
        data_dir (str): directory containing MR images to be normalized
        csf_mask (str): path to csf mask for data in data_dir
        contrast (str): contrast of MR images to be normalized (T1, T2, FLAIR or PD)
        mask_dir (str): if images are not skull-stripped, then provide brain mask
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        write_to_disk (bool): write the normalized data to disk or nah

    Returns:
        normalized (np.ndarray): set of normalized images from data_dir

    References:
        [1] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu,
            and R. T. Shinohara, “Removing inter-subject technical variability
            in magnetic resonance imaging studies,” Neuroimage, vol. 132,
            pp. 198–212, 2016.
    """
    data = glob(os.path.join(data_dir, '*.nii*'))
    input_files = StrVector(data)
    if mask_dir is None:
        mask_files = NULL
    else:
        masks = glob(os.path.join(mask_dir, '*.nii*'))
        if len(data) != len(masks):
            NormalizationError('Number of images and masks must be equal, Images: {}, Masks: {}'
                               .format(len(data), len(masks)))
        mask_files = StrVector(masks)
    if output_dir is None:
        output_files = NULL
    else:
        out_fns = []
        for fn in data:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base, ext))
        output_files = StrVector(out_fns)
    normalizedR = ravel.normalizeRAVEL(input_files, control_mask=csf_mask, output_files=output_files, brain_mask=mask_files,
                                       type=contrast, writeToDisk=write_to_disk, returnMatrix=True, verbose=False)
    normalized = np.array(normalizedR)
    return normalized


def csf_mask_intersection(data_dir, mask_dir=None, prob=0.9):
    """
    use all nifti T1w images in data_dir to create csf mask in common areas

    Args:
        data_dir (str): directory containing MR images to be normalized
        mask_dir (str): if images are not skull-stripped, then provide brain mask
        prob (float): given all data, proportion of data labeled as csf to be
            used for intersection

    Returns:
        intersection (np.ndarray): binary mask of common csf areas for all provided imgs
    """
    if not (0 <= prob <= 1):
        raise NormalizationError('prob must be between 0 and 1. {} given.'.format(prob))
    data = glob(os.path.join(data_dir, '*.nii*'))
    if mask_dir is None:
        masks = [None] * len(data)
    else:
        masks = glob(os.path.join(mask_dir, '*.nii*'))
    csf = [csf_mask(img, brain_mask=mask) for img, mask in zip(data, masks)]
    csf_sum = reduce(add, csf)
    intersection = np.zeros(csf_sum.shape)
    intersection[csf_sum > np.floor(len(data) * prob)] = 1
    return intersection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=True)
    parser.add_argument('--csf-mask', type=str, required=True)
    parser.add_argument('-c', '--contrast', type=str, default='T1')
    parser.add_argument('-m', '--mask_dir', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        if not os.path.isfile(args.csf_mask):
            csf_mask_inter = csf_mask_intersection(args.data_dir, args.mask_dir)
            nib.Nifti1Image(csf_mask_inter, None).to_filename(args.save_csf_mask)
        _ = ravel_normalize(args.data_dir, args.csf_mask, args.contrast, args.mask_dir, args.output_dir)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
