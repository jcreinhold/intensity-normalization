#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.exec.coregister

rigidly register a set of images to a template image (e.g., T1-w or MNI template)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 19, 2018
"""

import argparse
from glob import glob
import logging
import os
import shutil
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import ants
    from intensity_normalization.errors import NormalizationError
    from intensity_normalization.utilities.io import glob_nii, split_filename


def arg_parser():
    parser = argparse.ArgumentParser(description='coregister a set of MR images (e.g., to MNI or to the T1 image)')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    required.add_argument('-o', '--output-dir', type=str, required=True,
                        help='directory to output the corresponding registered img files')

    options = parser.add_argument_group('Options')
    options.add_argument('-t', '--template-dir', type=str, default=None,
                        help='directory of images to co-register the images to (if not provided, coreg to MNI)')
    options.add_argument('-tfm', '--tfm-dir', type=str, default=None,
                         help='directory of transformation files to register the images to (if not provided, will create tfm)')
    options.add_argument('--orientation', type=str, default='RAI',
                        help='output orientation of imgs')
    options.add_argument('-r', '--registration', type=str, default='Affine',
                        help='Use this type of registration (see ANTsPy for details) [Default: Affine]')
    options.add_argument('-itp', '--interpolator', type=str, default='bSpline',
                         help='Use this type of interpolation (see ANTsPy for details) [Default: bSpline]')
    options.add_argument('--no-rigid', action='store_true', default=False,
                        help='do not do rigid registration first')
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

        img_fns = glob_nii(args.img_dir)
        if not os.path.exists(args.output_dir):
            logger.info('Making Output Directory: {}'.format(args.output_dir))
            os.mkdir(args.output_dir)
        if not os.path.exists(os.path.join(args.output_dir, 'tfms')) and args.tfm_dir is None:
            logger.info('Making Transform Directory: {}'.format(args.output_dir))
            os.mkdir(os.path.join(args.output_dir, 'tfms'))
        if args.template_dir is None:
            logger.info('Registering image to MNI template')
            template = ants.image_read(ants.get_ants_data('mni')).reorient_image2(args.orientation)
            orientation = args.orientation
        else:
            template_fns = glob_nii(args.template_dir)
            if len(template_fns) != len(img_fns):
                raise NormalizationError('If template images are provided, they must be in '
                                         'correspondence (i.e., equal number) with the source images')
        if args.tfm_dir is not None:
            tfm_fns = sorted(glob(os.path.join(args.tfm_dir, '*.mat')))
            if len(tfm_fns) != len(img_fns):
                raise NormalizationError('If transforms are provided, they must be in '
                                         'correspondence (i.e., equal number) with the source images')
        for i, img in enumerate(img_fns):
            _, base, _ = split_filename(img)
            logger.info('Registering image to template: {} ({:d}/{:d})'.format(base, i+1, len(img_fns)))
            if args.template_dir is not None:
                template = ants.image_read(template_fns[i])
                orientation = template.orientation if hasattr(template, 'orientation') else None
            input_img = ants.image_read(img)
            input_img = input_img.reorient_image2(orientation) if orientation is not None else input_img
            if args.tfm_dir is None:
                if not args.no_rigid:
                    logger.info('Starting rigid registration: {} ({:d}/{:d})'.format(base, i+1, len(img_fns)))
                    mytx = ants.registration(fixed=template, moving=input_img, type_of_transform="Rigid")
                    tx = mytx['fwdtransforms'][0]
                else:
                    tx = None
                logger.info('Starting {} registration: {} ({:d}/{:d})'.format(args.registration, base, i+1, len(img_fns)))
                mytx = ants.registration(fixed=template, moving=input_img, initial_transform=tx, type_of_transform=args.registration)
            else:
                logger.info('Loading transform: {} ({:d}/{:d})'.format(tfm_fns[i], i+1, len(img_fns)))
                mytx = {'fwdtransforms': [tfm_fns[i]]}
            logger.debug(mytx)
            moved = ants.apply_transforms(template, input_img, mytx['fwdtransforms'], interpolator=args.interpolator)
            registered = os.path.join(args.output_dir, base + '_reg.nii.gz')
            ants.image_write(moved, registered)
            if args.tfm_dir is None:
                for j, tfm in enumerate(mytx['fwdtransforms']):
                    tfm_fn = os.path.join(args.output_dir, 'tfms', base + '_{}.mat'.format(j))
                    shutil.copy2(tfm, tfm_fn)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
