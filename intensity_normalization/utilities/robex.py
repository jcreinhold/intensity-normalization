#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.robex

a python wrapper for an R wrapper of ROBEX
used for rough, but always reasonably good, skull-stripping
of T1-w MR images

References:
    ﻿[1] J. E. Iglesias, C. Y. Liu, P. M. Thompson, and Z. Tu,
         “Robust brain extraction across datasets and comparison
         with publicly available methods,” IEEE Trans. Med. Imaging,
         vol. 30, no. 9, pp. 1617–1634, 2011.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 21, 2018
"""

import os
import warnings

import ants
from rpy2.robjects.packages import importr

from intensity_normalization.utilities.io import split_filename

ROBEX = importr('robex')


def robex(img, out_mask, skull_stripped=False):
    """
    perform skull-stripping on the registered image using the
    ROBEX algorithm

    Args:
        img (str): path to image to skull strip
        out_mask (str): path to output mask file
        skull_stripped (bool): return the mask
            AND the skull-stripped image [default = False]

    Returns:
        mask (ants.ANTsImage): mask/skull-stripped image
    """

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        _ = ROBEX.robex(img, outfile=out_mask)
    skull_stripped_img = ants.image_read(out_mask)
    mask = skull_stripped_img.get_mask(low_thresh=1)
    ants.image_write(mask, out_mask)
    if skull_stripped:
        # write the skull-stripped image to disk if desired (in addition to mask)
        dirname, base, _ = split_filename(out_mask)
        base = base.replace('mask', 'stripped') if 'mask' in base else base + '_stripped'
        ants.image_write(skull_stripped_img, os.path.join(dirname, base + '.nii.gz'))
    return mask
