#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.norm_quality

create a plot measuring the quality/consistency of
normalization given a directory of images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 04, 2018
"""

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from intensity_normalization.plot.quality import plot_pairwise_jsd


def arg_parser():
    parser = argparse.ArgumentParser(description='Create a plot measuring the quality/consistency of a normalization '
                                                 'method on a set of images given a directory of images. To measure'
                                                 'consistency, we create a histogram of the pairwise Jensen-Shannon '
                                                 'Divergence of all images and report some statistics of the hist.')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    required.add_argument('-m', '--mask-dir', type=str, default=None,
                        help='directory to brain masks for imgs')
    required.add_argument('-o', '--out-name', type=str, default='pairwisejsd.png',
                        help='name for output histogram (default: pairwisejsd.png)')

    options = parser.add_argument_group('Options')
    options.add_argument('--nbins', type=int, default=200,
                        help='number of bins to use when calculating JSD')
    options.add_argument('--fit-chi2', action='store_true', default=False,
                        help='fit a chi-square distribution to the data and report statistics')
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

        _ = plot_pairwise_jsd(args.img_dir, args.mask_dir, args.out_name, args.nbins, args.fit_chi2)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
