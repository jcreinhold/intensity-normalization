#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.robex.py

a python wrapper for an R wrapper of ROBEX
for robust skull-stripping (of T1w MR images)

References:
    ﻿[1] J. E. Iglesias, C. Y. Liu, P. M. Thompson, and Z. Tu,
         “Robust brain extraction across datasets and comparison
         with publicly available methods,” IEEE Trans. Med. Imaging,
         vol. 30, no. 9, pp. 1617–1634, 2011.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 21, 2018
"""

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from intensity_normalization.utilities.robex import robex
    from intensity_normalization.utilities.io import glob_nii, split_filename


def arg_parser():
    parser = argparse.ArgumentParser(description='Use ROBEX to skull-strip a set of nifti MR images')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed '
                             '(should all be T1w contrast)')
    parser.add_argument('-m', '--mask-dir', type=str, required=True,
                        help='directory to output the corresponding img files')
    parser.add_argument('-s', '--return-skull-stripped', action='store_true', default=False,
                        help='return skull-stripped images in addition to the masks')
    parser.add_argument('-v', '--verbosity', action="count", default=0,
                        help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
    if args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(__name__)
    try:
        img_fns = glob_nii(args.img_dir)
        if not os.path.exists(args.mask_dir):
            logger.info('Making Output Mask Directory: {}'.format(args.mask_dir))
            os.mkdir(args.mask_dir)
        for i, img in enumerate(img_fns, 1):
            _, base, _ = split_filename(img)
            logger.info('Creating Mask for Image: {}, ({:d}/{:d})'.format(base, i, len(img_fns)))
            mask = os.path.join(args.mask_dir, base + '_mask.nii.gz')
            _ = robex(os.path.abspath(img), os.path.abspath(mask), args.return_skull_stripped)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
