#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.whitestripe

Use the White Stripe method outlined in [1] to normalize
the intensity of an MR image

Note that this package requires RAVEL (and its dependencies)
to be installed in R

References:
﻿   [1] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee,
        F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
        D. S. Reich, and C. M. Crainiceanu, “Statistical normalization
        techniques for magnetic resonance imaging,” NeuroImage Clin.,
        vol. 6, pp. 9–19, 2014.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Apr 27, 2018
"""

from __future__ import print_function, division

import argparse
import sys

import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr

from intensity_normalization.errors import NormalizationError

fslr = importr('fslr')
ravel = importr('RAVEL')


def ws_normalize(img, brain_mask, contrast):
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
    pass

if __name__ == "__main__":
    sys.exit(main())
