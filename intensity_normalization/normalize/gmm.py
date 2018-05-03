#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.gmm

fit three gaussians to the histogram of
skull-stripped image and normalize the WM mean
to some standard value

Author: Blake Dewey
        Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Apr 24, 2018
"""

from __future__ import print_function, division

import argparse
import logging

import os
import sys

import nibabel as nib
import numpy as np
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    from sklearn.mixture import GMM as GaussianMixture

from intensity_normalization.utilities import io
from intensity_normalization.utilities.mask import gmm_class_mask, background_mask

logger = logging.getLogger()


def gmm_normalize(img, brain_mask=None, norm_value=1000, contrast='t1', bg_mask=None, wm_peak=None):
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
    norm_data = img_data/wm_peak*norm_value
    norm_data[norm_data < 0.1] = 0.0
    
    if bg_mask is not None:
        logger.info('Applying background mask...')
        masked_image = norm_data * bg_mask.get_data()
    else:
        masked_image = norm_data

    normalized = nib.Nifti1Image(masked_image, img.affine, img.header)
    return normalized
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True)
    parser.add_argument('-m', '--mask', type=str)
    parser.add_argument('-b', '--background-mask', type=str)
    parser.add_argument('-w', '--wm-peak', default=str)
    parser.add_argument('--save-wm-peak', action='store_true', default=False)
    parser.add_argument('--find-background-mask', action='store_true', default=False)
    parser.add_argument('--norm-value', type=float, default=1000)
    parser.add_argument('--contrast', type=str, choices=['t1', 't2'], default='t1')
    parser.add_argument('--keep-bg', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        img = io.open_nii(args.image)
        mask = io.open_nii(args.brain_mask)
        dirname, base, ext = io.split_filename(args.image)
        if args.find_background_mask:
            bg_mask = background_mask(img)
            bgfile = os.path.join(dirname, base + '_bgmask' + ext)
            io.save_nii(bg_mask, bgfile, is_nii=True)
        if args.wm_peak is not None:
            logger.info('Loading WM peak: ', args.wm_peak)
            peak = float(np.load(args.wm_peak))
        else:
            peak = gmm_class_mask(img, brain_mask=mask, contrast=args.contrast)
            if args.save_wm_peak:
                np.save(os.path.join(dirname, base + '_wmpeak.npy'), peak)
        normalized = gmm_normalize(img, mask, args.norm_value, args.contrast,
                                   args.background_mask, peak)
        outfile = os.path.join(dirname, base + '_norm' + ext)
        io.save_nii(normalized, outfile, is_nii=True)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
