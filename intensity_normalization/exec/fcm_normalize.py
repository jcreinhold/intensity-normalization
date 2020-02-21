#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.fcm_normalize

command line executable for fcm intensity normalization routine

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
    from intensity_normalization.normalize import fcm
    from intensity_normalization.utilities import io


def arg_parser():
    parser = argparse.ArgumentParser(description='Use FCM to model the tissue classes of the brain and use the '
                                                 'found tissue mean to normalize NIfTI MR images')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--image', type=str, required=True,
                          help='path to a directory of/single nifti MR image of the brain')
    required.add_argument('-m', '--brain-mask', type=str, default=None,
                          help='path to a directory of/single nifti brain mask for the image, '
                               'provide this if not providing WM mask, (step 1)')
    required.add_argument('-tm', '--tissue-mask', type=str, default=None,
                          help='path to a nifti mask of the tissue (found through FCM), '
                               'provide this if not providing the brain mask (step 2)')
    required.add_argument('-o', '--output-dir', type=str, default=None,
                          help='path to output normalized images '
                               '(default: to directory containing images in single img, '
                               'otherwise creates directory in cwd called fcm)')

    options = parser.add_argument_group('Options')
    options.add_argument('-c', '--contrast', type=str, default='t1',
                         help='contrast of images being normalized (must be `t1` when calculating the WM masks!)')
    options.add_argument('-tt', '--tissue-type', type=str, default='wm', choices=('wm', 'gm', 'csf'),
                         help='tissue type to be used for normalization')
    options.add_argument('-n', '--norm-value', type=float, default=1,
                         help='normalize the WM of the image to this value (default 1)')
    options.add_argument('-th', '--threshold', type=float, default=0.8,
                         help='threshold of FCM probability, used to determine tissue type in FCM result (default 0.8)')
    options.add_argument('-s', '--single-img', action='store_true', default=False,
                         help='image and mask are individual images, not directories')
    options.add_argument('-p', '--plot-hist', action='store_true', default=False,
                         help='plot the histograms of the normalized images, save it in the output directory')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def process(image_fn, brain_mask_fn, tissue_mask_fn, output_dir, args, logger):
    img = io.open_nii(image_fn)
    dirname, base, _ = io.split_filename(image_fn)
    if output_dir is not None:
        dirname = output_dir
        if not os.path.exists(dirname):
            logger.info('Making output directory: {}'.format(dirname))
            os.mkdir(dirname)
    if brain_mask_fn is not None:
        mask = io.open_nii(brain_mask_fn)
        tissue_mask = fcm.find_tissue_mask(img, mask, threshold=args.threshold, tissue_type=args.tissue_type)
        outfile = os.path.join(dirname, base + '_{}_mask.nii.gz'.format(args.tissue_type))
        io.save_nii(tissue_mask, outfile, is_nii=True)
    if tissue_mask_fn is not None:
        tissue_mask = io.open_nii(tissue_mask_fn)
        normalized = fcm.fcm_normalize(img, tissue_mask, args.norm_value)
        outfile = os.path.join(dirname, base + '_fcm.nii.gz')
        logger.info('Normalized image saved: {}'.format(outfile))
        io.save_nii(normalized, outfile, is_nii=True)


def main(args=None):
    args = arg_parser().parse_args(args)
    if not (args.brain_mask is None) ^ (args.tissue_mask is None):
        raise NormalizationError('Only one of {brain mask, tissue mask} should be given')
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
            if not os.path.isdir(args.image) or (False if args.brain_mask is None else not os.path.isdir(args.brain_mask)):
                raise NormalizationError('if single-img option off, then image and brain-mask must be directories')
            img_fns = io.glob_nii(args.image)
            mask_fns = io.glob_nii(args.brain_mask) if args.brain_mask is not None else [None] * len(img_fns)
            if len(img_fns) != len(mask_fns) and len(img_fns) > 0:
                raise NormalizationError('input images and masks must be in correspondence and greater than zero '
                                         '({:d} != {:d})'.format(len(img_fns), len(mask_fns)))
            args.output_dir = args.output_dir or 'fcm'
            output_dir_base = os.path.abspath(os.path.join(args.output_dir, '..'))

            if args.contrast.lower() == 't1' and args.tissue_mask is None:
                tissue_mask_dir = os.path.join(output_dir_base, 'tissue_masks')
                if os.path.exists(tissue_mask_dir):
                    logger.warning('Tissue mask directory already exists, may overwrite existing tissue masks!')
                else:
                    logger.info('Creating tissue mask directory: {}'.format(tissue_mask_dir))
                    os.mkdir(tissue_mask_dir)
                for i, (img, mask) in enumerate(zip(img_fns, mask_fns), 1):
                    _, base, _ = io.split_filename(img)
                    _, mask_base, _ = io.split_filename(mask)
                    logger.info('Creating tissue mask for {} ({:d}/{:d})'.format(base, i, len(img_fns)))
                    logger.debug('Tissue mask {} ({:d}/{:d})'.format(mask_base, i, len(img_fns)))
                    process(img, mask, None, tissue_mask_dir, args, logger)
            elif os.path.exists(args.tissue_mask):
                tissue_mask_dir = args.tissue_mask
            else:
                raise NormalizationError('If contrast is not t1, then tissue mask directory ({}) '
                                         'must already be created!'.format(args.tissue_mask))

            tissue_masks = io.glob_nii(tissue_mask_dir)
            for i, (img, tissue_mask) in enumerate(zip(img_fns, tissue_masks), 1):
                dirname, base, _ = io.split_filename(img)
                _, tissue_base, _ = io.split_filename(tissue_mask)
                logger.info('Normalizing image {} ({:d}/{:d})'.format(base, i, len(img_fns)))
                logger.debug('Tissue mask {} ({:d}/{:d})'.format(tissue_base, i, len(img_fns)))
                if args.output_dir is not None:
                    dirname = args.output_dir
                process(img, None, tissue_mask, dirname, args, logger)

        else:
            if not os.path.isfile(args.image):
                raise NormalizationError('if single-img option on, then image must be a file')
            if args.tissue_mask is None and args.contrast.lower() == 't1':
                logger.info('Creating tissue mask for {}'.format(args.image))
                process(args.image, args.brain_mask, None, args.output_dir, args, logger)
            elif os.path.isfile(args.tissue_mask):
                pass
            else:
                raise NormalizationError('If contrast is not t1, then tissue mask must be provided!')
            logger.info('Normalizing image {}'.format(args.image))
            dirname, base, _ = io.split_filename(args.image)
            dirname = args.output_dir or dirname
            if args.tissue_mask is None:
                tissue_mask = os.path.join(dirname, base + '_{}_mask.nii.gz'.format(args.tissue_type))
            else:
                tissue_mask = args.tissue_mask
            process(args.image, args.brain_mask, tissue_mask, dirname, args, logger)

        if args.plot_hist:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                from intensity_normalization.plot.hist import all_hists
                import matplotlib.pyplot as plt
            ax = all_hists(args.output_dir, args.brain_mask)
            ax.set_title('Fuzzy C-Means')
            plt.savefig(os.path.join(args.output_dir, 'hist.png'))

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
