#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.hm

Use the method of Nyul and Udupa [1] (updated in [2])
to do histogram matching intensity normalization on a
population of MR images

Note that this package requires RAVEL (and its dependencies)
to be installed in R

References:
    [1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image
        Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081,
        1999.
    [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold,
        D. L. Collins, and T. Arbel, “Evaluating intensity
        normalization on MRIs of human brain with multiple sclerosis,”
        Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 01, 2018
"""

from __future__ import print_function, division

import argparse
from glob import glob
import logging
import os
import sys

import numpy as np
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from rpy2.rinterface import NULL

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import io

ravel = importr('RAVEL')

logger = logging.getLogger()


def hm_normalize(data_dir, contrast, mask_dir=None, output_dir=None, write_to_disk=True):
    """
    Use histogram matching method ([1,2]) to normalize the intensities of a set of MR images

    Args:
        data_dir (str): directory containing MR images to be normalized
        contrast (str): contrast of MR images to be normalized (T1, T2, FLAIR or PD)
        mask_dir (str): if images are not skull-stripped, then provide brain mask
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        write_to_disk (bool): write the normalized data to disk or nah

    Returns:
        normalized (np.ndarray): set of normalized images from data_dir

    References:
        [1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image
            Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081,
            1999.
        [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold,
            D. L. Collins, and T. Arbel, “Evaluating intensity
            normalization on MRIs of human brain with multiple sclerosis,”
            Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.
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
    normalizedR = ravel.normalizeHM(input_files, output_files=output_files, brain_mask=mask_files,
                                    type=contrast, writeToDisk=write_to_disk, returnMatrix=True, verbose=False)
    normalized = np.array(normalizedR)
    return normalized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=True)
    parser.add_argument('-c', '--contrast', type=str, default='T1')
    parser.add_argument('-m', '--mask_dir', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        _ = hm_normalize(args.data_dir, args.contrast, args.mask_dir, args.output_dir)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
