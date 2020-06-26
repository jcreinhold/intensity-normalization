#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.zscore_normalize

command line executable for Z-score intensity normalization routine

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 30, 2018
"""

from __future__ import print_function, division

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from intensity_normalization.errors import NormalizationError
    from intensity_normalization.normalize import zscore
    from intensity_normalization.utilities import io


def arg_parser():
    parser = argparse.ArgumentParser(description='Normalize NIfTI MR image intensity by subtracting the mean '
                                                 'and dividing by the standard deviation of the whole brain')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--image', type=str, required=True,
                          help='path to a directory of/single nifti MR image of the brain')
    required.add_argument('-o', '--output-dir', type=str, default=None,
                          help='path to output normalized images '
                               '(default: to directory containing images)')

    options = parser.add_argument_group('Options')
    options.add_argument('-m', '--brain-mask', type=str, default=None,
                         help='path to a directory of/single nifti brain mask for the image. '
                              'use `none` for whole image z-score normalization.')
    options.add_argument('-s', '--single-img', action='store_true', default=False,
                         help='image and mask are individual images, not directories')
    options.add_argument('-p', '--plot-hist', action='store_true', default=False,
                         help='plot the histograms of the normalized images, save it in the output directory')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def process(image_fn, brain_mask_fn, output_dir, logger):
    img = io.open_nii(image_fn)
    dirname, base, _ = io.split_filename(image_fn)
    if output_dir is not None:
        dirname = output_dir
        if not os.path.exists(dirname):
            logger.info('Making output directory: {}'.format(dirname))
            os.mkdir(dirname)
    if brain_mask_fn is None:
        mask = None
    else:
        if brain_mask_fn == 'nomask':
            mask = 'nomask'
        else:
            mask = io.open_nii(brain_mask_fn)
    normalized = zscore.zscore_normalize(img, mask)
    outfile = os.path.join(dirname, base + '_zscore.nii.gz')
    logger.info('Normalized image saved: {}'.format(outfile))
    io.save_nii(normalized, outfile, is_nii=True)


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
        if not args.single_img:
            if not os.path.isdir(args.image):
                raise NormalizationError('if single-img option off, then image must be a directory')
            img_fns = io.glob_nii(args.image)
            if args.brain_mask is None:
                mask_fns = [None] * len(img_fns)
            else:
                if os.path.isdir(args.brain_mask):
                    mask_fns = io.glob_nii(args.brain_mask)
                else:
                    logger.info('whole image z-score normalization enabled')
                    mask_fns = ['nomask'] * len(img_fns)
            if len(img_fns) != len(mask_fns) and len(img_fns) > 0:
                raise NormalizationError('input images and masks must be in correspondence and greater than zero '
                                         '({:d} != {:d})'.format(len(img_fns), len(mask_fns)))

            for i, (img, mask) in enumerate(zip(img_fns, mask_fns), 1):
                logger.info('Normalizing image {} ({:d}/{:d})'.format(img, i, len(img_fns)))
                dirname, base, _ = io.split_filename(img)
                if args.output_dir is not None:
                    dirname = args.output_dir
                process(img, mask, dirname, logger)

        else:
            if not os.path.isfile(args.image):
                raise NormalizationError('if single-img option on, then image must be a file')
            logger.info('Normalizing image {}'.format(args.image))
            dirname, base, _ = io.split_filename(args.image)
            if args.output_dir is not None:
                dirname = args.output_dir
            process(args.image, args.brain_mask, dirname, logger)

        if args.plot_hist:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                from intensity_normalization.plot.hist import all_hists
                import matplotlib.pyplot as plt
            bm = args.brain_mask if args.brain_mask is None else \
                 args.brain_mask if os.path.isdir(args.brain_mask) else None
            ax = all_hists(args.output_dir, bm)
            ax.set_title('Z-Score')
            plt.savefig(os.path.join(args.output_dir, 'hist.png'))

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
