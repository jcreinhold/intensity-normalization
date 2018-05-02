#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.ravel


Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Apr 27, 2018
"""

import argparse
import sys

import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr

from intensity_normalization.errors import NormalizationError

fslr = importr('fslr')
ravel = importr('RAVEL')


def ravel_normalize(img, brain_mask, contrast):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--brain-mask', type=str)
    parser.add_argument('--norm-value', type=float, default=1000)
    args = parser.parse_args()
    if not (args.brain_mask is None) ^ (args.wm_mask is None):
        raise NormalizationError('Only one of {brain mask, wm mask} should be given')
    return args


def main():
    #ravel.maskIntersect(robjects.list("csf_mask1.nii.gz", "csf_mask2.nii.gz", "csf_mask3.nii.gz"), output.file="intersection_mask.nii.gz", prob=0.9)
    pass


if __name__ == "__main__":
    sys.exit(main())