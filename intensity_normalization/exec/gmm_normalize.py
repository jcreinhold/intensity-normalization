#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.gmm_normalize

command line executable for gmm intensity normalization routine

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 08, 2018
"""

from __future__ import print_function, division

import argparse
import logging
import os
import sys
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from intensity_normalization.errors import NormalizationError
    from intensity_normalization.normalize import gmm
    from intensity_normalization.utilities import io
    from intensity_normalization.utilities.mask import gmm_class_mask, background_mask


def arg_parser():
    parser = argparse.ArgumentParser(description='Use GMM to model the tissue classes in brain and '
                                                 'normalize the WM peak w/ this method for NIfTI MR images')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--image', type=str, required=True,
                          help='path to a nifti MR image of the brain')
    required.add_argument('-m', '--brain-mask', type=str,
                          help='path to a nifti brain mask for the image, '
                               'provide this if image is not skull-stripped')
    required.add_argument('-o', '--output-dir', type=str, default=None,
                          help='path to output normalized images '
                               '(default: to directory containing images)')

    options = parser.add_argument_group('Options')
    options.add_argument('-c', '--contrast', type=str, choices=['t1', 't2', 'flair'], default='t1',
                         help='pick the contrast of the input MR image (t1, t2, or flair)')
    options.add_argument('-b', '--background-mask', type=str,
                         help='path to a mask of the background')
    options.add_argument('-w', '--wm-peak', type=str, default=None,
                         help='saved WM peak value, found by first input T1 image,'
                              'can be used consecutively for other contrasts for the same patient')
    options.add_argument('-n', '--norm-value', type=float, default=1,
                         help='value by which to normalize the WM peak')
    options.add_argument('-s', '--single-img', action='store_true', default=False,
                         help='image and mask are individual images, not directories')
    options.add_argument('-p', '--plot-hist', action='store_true', default=False,
                         help='plot the histograms of the normalized images, save it in the output directory')
    options.add_argument('--save-wm-peak', action='store_true', default=False,
                         help='store the found WM peak, uses wm-peak as the name if true')
    options.add_argument('--find-background-mask', action='store_true', default=False,
                         help='calculate a mask for the background (to zero it out)')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def process(image, brain_mask, args, logger):
    img = io.open_nii(image)
    mask = io.open_nii(brain_mask)
    dirname, base, ext = io.split_filename(image)
    if args.output_dir is not None:
        dirname = args.output_dir
        if not os.path.exists(dirname):
            logger.info('Making output directory: {}'.format(dirname))
            os.mkdir(dirname)
    if args.find_background_mask:
        bg_mask = background_mask(img)
        bgfile = os.path.join(dirname, base + '_bgmask' + ext)
        io.save_nii(bg_mask, bgfile, is_nii=True)
    if args.wm_peak is not None:
        logger.info('Loading WM peak: ', args.wm_peak)
        peak = float(np.load(args.wm_peak))
    else:
        peak = gmm_class_mask(img, brain_mask=mask, contrast=args.contrast)
        if args.save_wm_peak:
            np.save(os.path.join(dirname, base + '_wmpeak.npy'), peak)
    normalized = gmm.gmm_normalize(img, mask, args.norm_value, args.contrast,
                                   args.background_mask, peak)
    outfile = os.path.join(dirname, base + '_gmm' + ext)
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
            if not os.path.isdir(args.image) or not os.path.isdir(args.brain_mask):
                raise NormalizationError('if single-img option off, then image and brain-mask must be directories')
            img_fns = io.glob_nii(args.image)
            mask_fns = io.glob_nii(args.brain_mask)
            if len(img_fns) != len(mask_fns) and len(img_fns) > 0:
                raise NormalizationError('input images and masks must be in correspondence and greater than zero '
                                         '({:d} != {:d})'.format(len(img_fns), len(mask_fns)))
            for i, (img, mask) in enumerate(zip(img_fns, mask_fns), 1):
                _, base, _ = io.split_filename(img)
                logger.info('Normalizing image {} ({:d}/{:d})'.format(img, i, len(img_fns)))
                process(img, mask, args, logger)
        else:
            if not os.path.isfile(args.image) or not os.path.isfile(args.brain_mask):
                raise NormalizationError('if single-img option on, then image and brain-mask must be files')
            process(args.image, args.brain_mask, args, logger)

        if args.plot_hist:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                from intensity_normalization.plot.hist import all_hists
                import matplotlib.pyplot as plt
            ax = all_hists(args.output_dir, args.brain_mask)
            ax.set_title('GMM')
            plt.savefig(os.path.join(args.output_dir, 'hist.png'))

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
