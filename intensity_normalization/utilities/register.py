#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalize.utilities.register

handles required registration for intensity normalization routines

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 08, 2018
"""

import gc
from glob import glob
import logging
import os

import ants

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities.io import split_filename

logger = logging.getLogger(__name__)


def register_to_template(img_dir, mask_dir=None, out_dir=None, tx_dir=None, template_img=0,
                         template_mask=0, reg_alg='SyNCC', **kwargs):
    """
    register a set of images using SyN (deformable registration)
    and write their output and transformations to disk

    Args:
        img_dir (str): directory containing MR images to be registered to a template
        mask_dir (str): directory containing MR images brain masks (optional)
        out_dir (str): directory to save images if you do not want them saved in
            a newly created directory (or existing dir) called `normalize_reg`
        tx_dir (str): directory to save registration tforms if you do not want them saved in
            a newly created directory (or existing dir) called `normalize_reg_tforms`
        template_img (int or str): number of img in img_dir, or a specified img path
            to be used as the template which all images are registered to
        template_mask (int or str): mask for template (used for better registration)
        reg_alg (str): registration algorithm to use, currently SyN w/ CC as metric
            (see ants.registration type_of_transform for more details/choices)
        kwargs: extra arguments for registration (see ants.registration for all available)

    Returns:
        None, writes registration transforms and registered images to disk
    """

    img_fns = sorted(glob(os.path.join(img_dir, '*.nii*')))
    if mask_dir is not None:
        mask_fns = sorted(glob(os.path.join(mask_dir, '*.nii*')))
    logger.debug('Input images: {}'.format(img_fns))

    if isinstance(template_img, int):
        template_img = img_fns[template_img]

    if (isinstance(template_mask, int) or isinstance(template_mask, str)) and mask_dir is not None:
        template_mask = mask_fns[template_mask]
        mask_fns = [fn for fn in mask_fns if fn != template_mask]
    elif not os.path.exists(template_mask):
        raise NormalizationError('Need to provide valid template mask name '
                                 '({}) if not providing brain masks'.format(template_mask))

    if tx_dir is None:
        tx_dir = os.path.join(os.getcwd(), 'reg_tforms')
        if os.path.exists(tx_dir):
            logger.warning('reg_tforms directory already exists, '
                           'may overwrite existing tforms!')
        else:
            os.mkdir(tx_dir)

    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'registered')
        if os.path.exists(out_dir):
            logger.warning('registered directory already exists, '
                           'may overwrite existing registered images!')
        else:
            os.mkdir(out_dir)

    img_fns = [fn for fn in img_fns if fn != template_img]
    template = ants.image_read(template_img)

    if template_mask is not None:
        template_mask = ants.image_read(template_mask)
    else:
        template_mask = None

    if mask_dir is not None:
        template = template * template_mask
        template_mask = None

    _, base, _ = split_filename(template_img)
    logger.debug('Template image: {}'.format(base))
    for i, fn in enumerate(img_fns, 1):
        _, base, _ = split_filename(fn)
        logger.debug('Image to register ({}): {}'.format(i, base))

    # control verbosity of output when making registration function call
    verbose = True if logger.getEffectiveLevel() == logging.getLevelName('DEBUG') else False

    for i, fn in enumerate(img_fns, 1):
        img = ants.image_read(fn)
        if mask_dir is not None:
            mask = ants.image_read(mask_fns[i-1])
            img = img * mask
        _, base, _ = split_filename(fn)
        logger.info('Registering image: {} ({:d}/{:d})'.format(base, i, len(img_fns)))
        reg_result = ants.registration(template, img, type_of_transform=reg_alg,
                                       mask=template_mask, verbose=verbose,
                                       outprefix=os.path.join(tx_dir, base + '_'),
                                       **kwargs)
        moved = ants.apply_transforms(fixed=template, moving=img, interpolator='bSpline',
                                      transformlist=reg_result['fwdtransforms'])
        moved_fn = os.path.join(out_dir, base + '_reg.nii.gz')
        logger.debug('Output registered image: {}'.format(moved_fn))
        ants.image_write(moved, moved_fn)
        # try to keep memory usage low w/ manual garbage collection
        del img, reg_result, moved
        gc.collect()


def unregister(reg_dir, tx_dir, template_img, out_dir=None):
    """
    undo the template registration process, this should be used with HM and RAVEL
    intensity normalization methods since they require the images to initially be
    deformably aligned

    Args:
        reg_dir (str): directory of registered (and normalized probably) nifti images
        tx_dir (str): directory to from which to load registration tforms
        out_dir (str): directory to save de-registered images if you do not want them saved in
            a newly created directory (or existing dir) called `normalized`
        template_img (int or str): a specified img path used as the template which all
            images were registered to

    Returns:
        None, writes de-registered images to disk
    """
    reg_fns = glob(os.path.join(reg_dir, '*.nii*'))
    reg_fns = sorted([fn for fn in reg_fns if template_img != fn])
    affine_fns = glob(os.path.join(tx_dir, '*.mat'))
    deformable_fns = glob(os.path.join(tx_dir, '*InverseWarp.nii.gz'))
    reg_func_fns = sorted(affine_fns + deformable_fns)
    template = ants.image_read(template_img)
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'normalized')
    if os.path.exists(out_dir):
        logger.warning('normalized directory already exists, '
                       'may overwrite existing normalized images!')
    else:
        os.mkdir(out_dir)

    for i, fn in enumerate(reg_fns):
        _, base, _ = split_filename(fn)
        img = ants.image_read(fn)
        transformlist = sorted([reg_fn for reg_fn in reg_func_fns if base in reg_fn])
        whichtoinvert = [True if '.mat' in fn else False for fn in transformlist]
        logger.info('De-registering image: {} ({:d}/{:d})'.format(base, i, len(reg_fns)))
        unmoved = ants.apply_transforms(fixed=template, moving=img, interpolator='bSpline',
                                        transformlist=transformlist, whichtoinvert=whichtoinvert)
        ants.image_write(unmoved, os.path.join(out_dir, base + '_norm.nii.gz'))
        # try to keep memory usage low w/ manual garbage collection
        del img
        gc.collect()
