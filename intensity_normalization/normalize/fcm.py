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
import numpy as np
from skfuzzy import cmeans

from utilities.io import split_filename

logger = logging.getLogger()


def fcm_normalize(img, brain_mask=None, wm_mask=None, norm_value=1000):
    if not (brain_mask is None) ^ (wm_mask is None):
        raise ValueError('Only one of {Mask, Peak-File} should be given')

    image = os.path.abspath(os.path.expanduser(img))
    if brain_mask is not None:
        brain_mask = os.path.abspath(os.path.expanduser(brain_mask))
    if wm_mask is not None:
        wm_mask = os.path.abspath(os.path.expanduser(wm_mask))

    dirname, base, _ = split_filename(image)
    obj = nib.load(image)
    img_data = obj.get_data()

    if wm_mask is not None:
        wm_mask = nib.load(wm_mask).get_data() > 0
    else:
        obj = nib.load(image)
        img_data = obj.get_data()
        mask_data = nib.load(brain_mask).get_data() > 0
        [t1_cntr, t1_mem, _, _, _, _, _] = cmeans(img_data[mask_data].reshape(-1, len(mask_data[mask_data])),
                                                  3, 2, 0.005, 50)
        t1_mem_list = [t1_mem[i] for i, _ in sorted(enumerate(t1_cntr), key=lambda x: x[1])]  # CSF/GM/WM
        t1_mem = np.zeros(img_data.shape + (3,))
        for i in range(3):
            t1_mem[..., i][mask_data] = t1_mem_list[i]
        wm_mask = t1_mem[..., 2] > 0.8
        nib.Nifti1Image(wm_mask, obj.affine, obj.header).to_filename(os.path.join(dirname, base + '_wmmask.nii.gz'))
    wm_mean = img_data[wm_mask].mean()
    nib.Nifti1Image(obj.get_data() / wm_mean * norm_value,
                    obj.affine, obj.header).to_filename(os.path.join(dirname, base + '_norm.nii.gz'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--brain-mask', type=str)
    parser.add_argument('--wm-mask', type=str)
    parser.add_argument('--norm-value', type=float, default=1000)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        fcm_normalize(args.image, args.brain_mask, args.wm_mask, args.norm_value)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
