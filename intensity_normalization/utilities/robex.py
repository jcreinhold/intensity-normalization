#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.robex

python wrapper for R wrapper of ROBEX
used for rough, but always reasonably good, skull-stripping

References:
    ﻿[1] J. E. Iglesias, C. Y. Liu, P. M. Thompson, and Z. Tu,
         “Robust brain extraction across datasets and comparison
         with publicly available methods,” IEEE Trans. Med. Imaging,
         vol. 30, no. 9, pp. 1617–1634, 2011.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 21, 2018
"""

import warnings

import ants
from rpy2.robjects.packages import importr

ROBEX = importr('robex')


def robex(img, out_mask):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        _ = ROBEX.robex(img, outfile=out_mask)
    mask = ants.image_read(out_mask)
    mask = mask.get_mask(low_thresh=1)
    ants.image_write(mask, out_mask)
    return mask
