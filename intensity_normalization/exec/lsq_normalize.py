#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.lsq_normalize

command line executable for least squares intensity normalization routine

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 08, 2018
"""

from __future__ import print_function, division

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from intensity_normalization.normalize import lsq


def arg_parser():
    parser = argparse.ArgumentParser(description='Use least squares intensity normalization '
                                                 'on a set of NIfTI MR images')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed '
                             '(should all be of one contrast)')
    required.add_argument('-o', '--output-dir', type=str, default=None,
                         help='save the normalized images to this path [Default = None]')

    options = parser.add_argument_group('Options')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                           help='directory for corresponding brain masks for img-dir (not intelligently sorted, '
                                'so ordering must be consistent in directory). [Default = None]')
    options.add_argument('-p', '--plot-hist', action='store_true', default=False,
                         help='plot the histograms of the normalized images, save it in the output directory')
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

        _ = lsq.lsq_normalize(args.img_dir, args.mask_dir, args.output_dir)

        if args.plot_hist:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                from intensity_normalization.plot.hist import all_hists
                import matplotlib.pyplot as plt
            ax = all_hists(args.output_dir, args.mask_dir)
            ax.set_title('Least Squares')
            plt.savefig(os.path.join(args.output_dir, 'hist.png'))

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
