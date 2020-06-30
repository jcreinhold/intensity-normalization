#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.preprocess

preprocess.py a set of MR images according to a simple scheme,
that is:
    1) N4 bias field correction
    2) resample to 1mm x 1mm x 1mm
    3) reorient images to RAI

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
    from intensity_normalization.utilities.preprocess import preprocess


def arg_parser():
    parser = argparse.ArgumentParser(description='Do some basic preprocessing on a set of NIfTI MR images of the brain. '
                                                 '(i.e., resampling, reorientation, and bias field correction)')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                          help='path to directory with images to be processed')
    required.add_argument('-o', '--out-dir', type=str, required=True,
                          help='output directory for preprocessed files')

    options = parser.add_argument_group('Options')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                          help='directory to output the corresponding mask files')
    options.add_argument('-r', '--resolution', nargs=3, type=float, default=None,
                         help='resolution for resampled images (if not set, then keep image resolution)')
    options.add_argument('--orientation', type=str, default='RAI',
                         help='orientation of preprocessed images')
    options.add_argument('--n4-opts', type=str, default=None,
                         help='n4 convergence options. Add arguments to json file or formatted string, e.g., '
                              "'{\"iters\": [200, 200, 200, 200], \"tol\": 0.0005}', "
                              'see ants.n4_bias_field_correction for details about options.')
    options.add_argument('-v', '--verbosity', action="count", default=0,
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
        if not os.path.isdir(args.img_dir):
            raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')
        if args.mask_dir is not None:
            if not os.path.isdir(args.mask_dir):
                raise ValueError('(-m / --mask-dir) argument needs to be a directory of NIfTI images.')

        if args.n4_opts is None:
            n4_opts = None
        else:
            import json
            n4_opts = json.loads(args.n4_opts)
        preprocess(args.img_dir, args.out_dir, args.mask_dir, args.resolution, args.orientation, n4_opts)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
