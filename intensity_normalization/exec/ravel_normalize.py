#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.ravel_normalize

command line executable for (modified) RAVEL intensity normalization routine

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 08, 2018
"""

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from intensity_normalization.errors import NormalizationError
    from intensity_normalization.normalize import ravel
    from intensity_normalization.utilities import io


def arg_parser():
    parser = argparse.ArgumentParser(description='Use RAVEL to normalize a directory of nifti MR images of the brain.')

    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed '
                             '(should all be of one contrast)')

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--output-dir', type=str, default=None,
                           help='save the normalized images to this path [Default = None]')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                           help='if images are not skull-stripped, directory for '
                                'corresponding brain masks for img-dir (not intelligently sorted, '
                                'so ordering must be consistent in directory) [Default = None]')
    options.add_argument('-c', '--contrast', type=str, default='t1', choices=['t1', 't2', 'flair'],
                           help='contrast of the images in img-dir, (e.g, t1, t2, or, flair.) [Default = t1]')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    options.add_argument('-p', '--plot-hist', action='store_true', default=False,
                         help='plot the histograms of the normalized images, save it in the output directory')

    adv_options = parser.add_argument_group('Advanced Options')
    adv_options.add_argument('-b', '--num-unwanted-factors', type=int, default=1,
                             help='number of unwanted factors to eliminate (see b in RAVEL paper) [Default = 1]')
    adv_options.add_argument('-t', '--control-membership-threshold', type=float, default=0.99,
                             help='threshold for the membership of the control (CSF) voxels [Default = 0.99]')
    adv_options.add_argument('-s', '--segmentation-smoothness', type=float, default=0.25,
                             help='smoothness parameter for segmentation for control voxels [Default = 0.25]')
    adv_options.add_argument('--use-fcm', action='store_true', default=False,
                             help='use fuzzy c-means segmentation instead of atropos')
    adv_options.add_argument('--no-whitestripe', action='store_false', default=True,
                             help='do not use whitestripe in RAVEL if this flag is on')
    adv_options.add_argument('--no-registration', action='store_false', default=True,
                             help='do not do deformable registration to find control mask '
                                  '(*much* slower but follows paper and is more consistent)')
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
        img_fns = io.glob_nii(args.img_dir)
        mask_fns = io.glob_nii(args.mask_dir)
        if len(img_fns) != len(mask_fns) or len(img_fns) == 0:
            raise NormalizationError('Image directory ({}) and mask directory ({}) must contain the same '
                                     '(positive) number of images!'.format(args.img_dir, args.mask_dir))

        logger.info('Normalizing the images according to RAVEL')
        Z, _ = ravel.ravel_normalize(args.img_dir, args.mask_dir, args.contrast, do_whitestripe=args.no_whitestripe,
                                     b=args.num_unwanted_factors, membership_thresh=args.control_membership_threshold,
                                     do_registration=args.no_registration, segmentation_smoothness=args.segmentation_smoothness,
                                     use_fcm=args.use_fcm)

        V = ravel.image_matrix(img_fns, args.contrast, masks=mask_fns)
        V_norm = ravel.ravel_correction(V, Z)
        normalized = ravel.image_matrix_to_images(V_norm, img_fns)

        # save the normalized images to disk
        output_dir = os.getcwd() if args.output_dir is None else args.output_dir
        out_fns = []
        for fn in img_fns:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for norm, out_fn in zip(normalized, out_fns):
            norm.to_filename(out_fn)

        if args.plot_hist:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                from intensity_normalization.plot.hist import all_hists
                import matplotlib.pyplot as plt
            ax = all_hists(output_dir, args.mask_dir)
            ax.set_title('RAVEL')
            plt.savefig(os.path.join(output_dir, 'hist.png'))

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
