#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.tissue_mask

create tissue masks/memberships using FCM or GMM
for skull-stripped brain or brain w/ mask

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 23, 2018
"""

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from intensity_normalization.utilities.mask import fcm_class_mask, gmm_class_mask
    from intensity_normalization.utilities import io


def arg_parser():
    parser = argparse.ArgumentParser(description='Create a tissue mask for a given brain (i.e., label CSF/GM/WM)')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                          help='path to directory with images to be processed '
                               '(should all be T1w contrast)')
    required.add_argument('-m', '--mask-dir', type=str, default=None,
                          help='directory of corresponding brain mask img files')
    required.add_argument('-o', '--output-dir', type=str, default=None,
                          help='directory to output the tissue masks img files')

    options = parser.add_argument_group('Options')
    options.add_argument('--gmm', action='store_true', default=False,
                         help='use a gmm to create tissue class membership instead of FCM')
    options.add_argument('--memberships', action='store_true', default=False,
                         help='output individual class membership masks instead of hard segmentation')
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

        img_fns = io.glob_nii(args.img_dir)
        if args.mask_dir is not None:
            mask_fns = io.glob_nii(args.mask_dir)
        else:
            mask_fns = [None] * len(img_fns)
        if not os.path.exists(args.output_dir):
            logger.info('Making Output Directory: {}'.format(args.output_dir))
            os.mkdir(args.output_dir)
        hard_seg = not args.memberships
        for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns), 1):
            _, base, _ = io.split_filename(img_fn)
            logger.info('Creating Mask for Image: {}, ({:d}/{:d})'.format(base, i, len(img_fns)))
            img = io.open_nii(img_fn)
            mask = io.open_nii(mask_fn)
            tm = fcm_class_mask(img, mask, hard_seg) if not args.gmm else gmm_class_mask(img, mask, 't1', False, hard_seg)
            tissue_mask = os.path.join(args.output_dir, base + '_tm')
            if args.memberships:
                classes = ('csf', 'gm', 'wm')
                for j, c in enumerate(classes):
                    io.save_nii(img, tissue_mask + '_' + c + '.nii.gz', tm[..., j])
            else:
                io.save_nii(img, tissue_mask + '.nii.gz', tm)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
