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

from intensity_normalization.utilities import io, mask
from intensity_normalization.errors import NormalizationError

logger = logging.getLogger()


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
    wm_mean = img_data[wm_mask].mean()
    normalized = nib.Nifti1Image((img.get_data() / wm_mean) * norm_value,
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
        wm_mask (np.ndarray): white matter mask for img
    """
    t1_mem = mask.fcm_class_mask(img, brain_mask)
    wm_mask = t1_mem[..., 2] > threshold
    return wm_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--brain-mask', type=str)
    parser.add_argument('--wm-mask', type=str)
    parser.add_argument('--norm-value', type=float, default=1000)
    args = parser.parse_args()
    if not (args.brain_mask is None) ^ (args.wm_mask is None):
        raise NormalizationError('Only one of {brain mask, wm mask} should be given')
    return args


def main():
    args = parse_args()
    try:
        img = io.open_nii(args.image)
        dirname, base, _ = io.split_filename(args.image)
        if args.brain_mask is not None:
            brain_mask = io.open_nii(args.brain_mask)
            wm_mask = find_wm_mask(img, brain_mask)
            outfile = os.path.join(dirname, base + '_wmmask.nii.gz')
            io.save_nii(img, outfile, data=wm_mask)
        if args.wm_mask is not None:
            wm_mask = io.open_nii(args.brain_mask)
            normalized = fcm_normalize(img, wm_mask, args.norm_value)
            outfile = os.path.join(dirname, base + '_norm.nii.gz')
            io.save_nii(normalized, outfile, is_nii=True)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
