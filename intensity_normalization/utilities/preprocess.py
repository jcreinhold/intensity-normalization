#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.preprocess

preprocess MR images according to a simple scheme,
that is:
    1) N4 bias field correction
    2) resample to 1mm x 1mm x 1mm
    3) reorient images to RAI
this process requires brain masks for all images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 21, 2018
"""

from glob import glob
import logging
import os

import ants

from intensity_normalization.utilities.io import split_filename

logger = logging.getLogger(__name__)


def preprocess(img_dir, mask_dir, out_dir, res=(1,1,1), orientation='RAI', n4_opts=None):
    """
    preprocess MR images according to a simple scheme,
    that is:
        1) N4 bias field correction
        2) resample to 1mm x 1mm x 1mm
        3) reorient images to RAI
    this process requires brain masks for all images

    Args:
        img_dir (str): path to directory containing images
        mask_dir (str): path to directory containing masks
        out_dir (str): path to directory for output preprocessed files
        res (tuple): resolution for resampling (default: (1,1,1) in mm
        n4_opts (dict): n4 processing options (default: None)

    Returns:
        None, outputs preprocessed images to file in given out_dir
    """

    if n4_opts is None:
        n4_opts = {'iters': [200, 200, 200, 200], 'tol': 0.0005}
    logger.debug('N4 Options are: {}'.format(n4_opts))

    # get and check the images and masks
    img_fns = sorted(glob(os.path.join(img_dir, '*.nii.gz')))
    mask_fns = sorted(glob(os.path.join(mask_dir, '*.nii.gz')))
    assert len(img_fns) == len(mask_fns), 'Number of images and masks must be equal ({:d} != {:d})'\
        .format(len(img_fns), len(mask_fns))

    # create the output directory structure
    out_img_dir = os.path.join(out_dir, 'imgs')
    out_mask_dir = os.path.join(out_dir, 'mask')
    if not os.path.exists(out_dir):
        logger.info('Making output directory structure: {}'.format(out_dir))
        os.mkdir(out_dir)
    if not os.path.exists(out_img_dir):
        logger.info('Making image output directory: {}'.format(out_img_dir))
        os.mkdir(out_img_dir)
    if not os.path.exists(out_mask_dir):
        logger.info('Making mask output directory: {}'.format(out_mask_dir))
        os.mkdir(out_mask_dir)

    # preprocess the images by n4 correction, resampling, and reorientation
    for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns), 1):
        _, img_base, img_ext = split_filename(img_fn)
        _, mask_base, mask_ext = split_filename(mask_fn)
        logger.info('Reading image: {} ({:d}/{:d})'.format(img_base, i, len(img_fns)))
        img = ants.image_read(img_fn)
        ones = ants.image_clone(img) * 0 + 1
        mask = ants.image_read(mask_fn)
        smoothed_mask = ants.smooth_image(mask, 1)
        img = ants.n4_bias_field_correction(img, ones, convergence=n4_opts)
        img = ants.n4_bias_field_correction(img, ones, convergence=n4_opts, weight_mask=smoothed_mask)
        img = ants.resample_image(img, res, False, 4)
        mask = ants.resample_image(mask, res, False, 1)
        img = img.reorient_image2(orientation)
        mask = mask.reorient_image2(orientation)
        logger.info('Writing preprocessed image: {} ({:d}/{:d})'.format(img_base, i, len(img_fns)))
        out_img = os.path.join(out_img_dir, img_base + img_ext)
        out_mask = os.path.join(out_mask_dir, mask_base + mask_ext)
        ants.image_write(img, out_img)
        ants.image_write(mask, out_mask)
