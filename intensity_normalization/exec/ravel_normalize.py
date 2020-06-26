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
    parser = argparse.ArgumentParser(description='Use RAVEL to normalize a directory of NIfTI MR images of the brain.')

    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed (should all be of one contrast)')
    required.add_argument('-m', '--mask-dir', type=str, required=True,
                           help='directory of corresponding brain masks for img-dir (not intelligently sorted, '
                                'so ordering must be consistent in directory, e.g., if img1.nii.gz and img2.nii.gz are '
                                'in img-dir, then the mask should preserve the alphabetical/numeric order like'
                                'naming them img1_mask.nii.gz and img2_mask.nii.gz)' )

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--output-dir', type=str, default=None,
                           help='save the normalized images to this path [Default = None]')
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
    adv_options.add_argument('--use-atropos', action='store_true', default=False,
                             help='use atropos instead of fuzzy c-means segmentation')
    adv_options.add_argument('--no-whitestripe', action='store_false', default=True,
                             help='do not use whitestripe in RAVEL if this flag is on')
    adv_options.add_argument('--no-registration', action='store_false', default=True,
                             help='do not do deformable registration to find control mask '
                                  '(*much* slower but follows paper and is more consistent)')
    adv_options.add_argument('--sparse-svd', action='store_true', default=False,
                             help='use a sparse version of the svd (should have lower memory requirements)')
    adv_options.add_argument('--csf-masks', action='store_true', default=False,
                             help='mask directory corresponds to csf masks instead of brain masks, '
                                  'assumes images are deformably co-registered')

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
        if not os.path.isdir(args.mask_dir):
            raise ValueError('(-m / --mask-dir) argument needs to be a directory of NIfTI images.')

        img_fns = io.glob_nii(args.img_dir)
        mask_fns = io.glob_nii(args.mask_dir)
        if len(img_fns) != len(mask_fns) or len(img_fns) == 0:
            raise NormalizationError('Image directory ({}) and mask directory ({}) must contain the same '
                                     '(positive) number of images!'.format(args.img_dir, args.mask_dir))

        logger.info('Normalizing the images according to RAVEL')
        Z, _ = ravel.ravel_normalize(args.img_dir, args.mask_dir, args.contrast, do_whitestripe=args.no_whitestripe,
                                     b=args.num_unwanted_factors, membership_thresh=args.control_membership_threshold,
                                     do_registration=args.no_registration, segmentation_smoothness=args.segmentation_smoothness,
                                     use_fcm=not args.use_atropos, sparse_svd=args.sparse_svd, csf_masks=args.csf_masks)

        V = ravel.image_matrix(img_fns, args.contrast, masks=mask_fns)
        V_norm = ravel.ravel_correction(V, Z)
        normalized = ravel.image_matrix_to_images(V_norm, img_fns)

        # save the normalized images to disk
        output_dir = os.getcwd() if args.output_dir is None else args.output_dir
        out_fns = []
        for fn in img_fns:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + '_ravel' + ext))
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
