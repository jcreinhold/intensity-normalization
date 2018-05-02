#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.hm


Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 01, 2018
"""

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
