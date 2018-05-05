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
import os
import sys

import numpy as np
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr

from intensity_normalization.errors import NormalizationError

fslr = importr('fslr')
ravel = importr('RAVEL')


def hm_normalize(data_dir, contrast):
    input_files = StrVector(glob(os.path.join(data_dir, '*.nii*')))
    ravel.normalizeHM()
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=True)
    parser.add_argument('-m', '--mask', type=str)
    args = parser.parse_args()
    return args


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
