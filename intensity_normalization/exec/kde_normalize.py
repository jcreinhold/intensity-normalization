#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.kde_normalize

command line executable for kernel density intensity normalization routine

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
    from intensity_normalization.errors import NormalizationError
    from intensity_normalization.normalize import kde
    from intensity_normalization.utilities import io


def arg_parser():
    parser = argparse.ArgumentParser(description='Use Kernel Density Estimation method to WM peak '
                                                 'normalize a set of NIfTI MR images.')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--image', type=str, required=True,
                        help='path to a nifti MR image of the brain')
    required.add_argument('-m', '--brain-mask', type=str, default=None,
                        help='path to a nifti brain mask for the image,'
                             'if image is not skull-stripped')

    options = parser.add_argument_group('Options')
    options.add_argument('-c', '--contrast', type=str, default='t1', choices=('t1','t2','flair','md','largest','first','last'),
                         help='contrast of the image (e.g., `t1`, `t2`, etc.)')
    options.add_argument('-o', '--output-dir', type=str, default=None,
                         help='path to output normalized images '
                              '(default: to directory containing images')
    options.add_argument('-n', '--norm-value', type=float, default=1,
                         help='value by which to normalize the WM peak, default 1')
    options.add_argument('-s','--single-img', action='store_true', default=False,
                         help='image and mask are individual images, not directories')
    options.add_argument('-p', '--plot-hist', action='store_true', default=False,
                         help='plot the histograms of the normalized images, save it in the output directory')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def process(image_fn, brain_mask_fn, args, logger):
    img = io.open_nii(image_fn)
    if args.brain_mask is not None:
        mask = io.open_nii(brain_mask_fn)
    else:
        mask = None
    dirname, base, _ = io.split_filename(image_fn)
    if args.output_dir is not None:
        dirname = args.output_dir
        if not os.path.exists(dirname):
            logger.info('Making output directory: {}'.format(dirname))
            os.mkdir(dirname)
    normalized = kde.kde_normalize(img, mask, args.contrast, args.norm_value)
    outfile = os.path.join(dirname, base + '_kde.nii.gz')
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
            ax.set_title('KDE')
            plt.savefig(os.path.join(args.output_dir, 'hist.png'))

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
