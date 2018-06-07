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
import numpy as np

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
    logger.debug('Input images: {}'.format(img_fns))
    if mask_dir is not None:
        mask_fns = sorted(glob(os.path.join(mask_dir, '*.nii*')))
        logger.debug('Input masks: {}'.format(mask_fns))
    else:
        mask_fns = [None] * len(img_fns)

    # (awfully) handle loading in template img and mask depending on different input
    # if template_img and template_mask not provided, then use MNI
    if template_img is None and template_mask is None:
        template_img = template_mask = ants.get_ants_data('mni')
    else:
        # can provide template img as an int and template mask as an int
        if isinstance(template_img, int) and isinstance(template_mask, int) and mask_dir is not None:
            template_img = img_fns[template_img]
            template_mask = mask_fns[template_mask]
        # can provide template img as a path and template mask as a path
        elif isinstance(template_img, str) and isinstance(template_mask, str):
            if not os.path.exists(template_img) or not os.path.exists(template_mask):
                raise NormalizationError('Need to provide valid template img/mask name')
        else:
            raise NormalizationError('Input Template image ({}) and mask ({}) invalid types'
                                     .format(template_img, template_mask))

    # make sure template image/mask not in set of images to register
    img_fns = [fn for fn in img_fns if fn != template_img]
    mask_fns = [fn for fn in mask_fns if fn != template_mask]
    template = ants.image_read(template_img)
    tmask = ants.image_read(template_mask)
    template = template * tmask

    # verify that the template image is correct and the images to register are good
    _, base, _ = split_filename(template_img)
    logger.debug('Template image: {}'.format(base))
    for i, fn in enumerate(img_fns, 1):
        _, base, _ = split_filename(fn)
        logger.debug('Image to register ({}): {}'.format(i, base))

    # format and create necessary directories
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

    # control verbosity of output when making registration function call
    verbose = True if logger.getEffectiveLevel() == logging.getLevelName('DEBUG') else False

    # actually do the registration here
    for i, fn in enumerate(img_fns):
        img = ants.image_read(fn)
        if mask_dir is not None:
            mask = ants.image_read(mask_fns[i])
            img = img * mask
        _, base, _ = split_filename(fn)
        logger.info('Registering image: {} ({:d}/{:d})'.format(base, i+1, len(img_fns)))
        reg_result = ants.registration(fixed=template, moving=img, type_of_transform=reg_alg,
                                       mask=tmask, verbose=verbose,
                                       outprefix=os.path.join(tx_dir, base + '_'),
                                       **kwargs)
        # TODO: apply_transforms does not currently work as expected, need to investigate why
        #moved = ants.apply_transforms(fixed=template, moving=img, interpolator='bSpline',
        #                              transformlist=reg_result['fwdtransforms'])
        moved = reg_result['warpedmovout']
        moved_fn = os.path.join(out_dir, base + '_reg.nii.gz')
        logger.debug('Output registered image: {}'.format(moved_fn))
        ants.image_write(moved, moved_fn)
        # try to keep memory usage low w/ manual garbage collection
        del img, reg_result, moved
        gc.collect()


def unregister(reg_dir, tx_dir, template_img, out_dir=None, mask_dir=None):
    """
    undo the template registration process, this should be used with HM and RAVEL
    intensity normalization methods since they require the images to initially be
    deformably aligned

    Args:
        reg_dir (str): directory of registered (and normalized probably) nifti images
        tx_dir (str): directory to from which to load registration tforms
        out_dir (str): directory to save de-registered images if you do not want them saved in
            a newly created directory (or existing dir) called `normalized`
        template_img (str): a specified img path used as the template which all
            images were registered to

    Returns:
        None, writes de-registered images to disk
    """
    reg_fns = glob(os.path.join(reg_dir, '*.nii*'))
    reg_fns = sorted([fn for fn in reg_fns if template_img != fn])
    if mask_dir is not None:
        _, template_base, _ = split_filename(template_img)
        mask_fns = glob(os.path.join(mask_dir, '*.nii*'))
        mask_fns = sorted([fn for fn in mask_fns if template_base not in fn])
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

    # control verbosity of output when making registration function call
    verbose = True if logger.getEffectiveLevel() == logging.getLevelName('DEBUG') else False

    for i, fn in enumerate(reg_fns):
        _, base, _ = split_filename(fn)
        img = ants.image_read(fn)
        transformlist = sorted([reg_fn for reg_fn in reg_func_fns if base.replace('_reg', '') in reg_fn])
        whichtoinvert = [True if '.mat' in fn else False for fn in transformlist]
        logger.info('De-registering image: {} ({:d}/{:d})'.format(base, i+1, len(reg_fns)))
        unmoved = ants.apply_transforms(fixed=template, moving=img, interpolator='linear',
                                        transformlist=transformlist, whichtoinvert=whichtoinvert,
                                        verbose=verbose)
        if mask_dir is not None:
            # if desired, include masks so that the zeros at the borders of the images are gone
            # the zeros come from the affine transformation
            mask = ants.image_read(mask_fns[i])
            unmoved_data = unmoved.numpy()
            minval = np.min(unmoved_data)
            unmoved_data[mask.numpy() == 0] = minval
            unmoved = unmoved.new_image_like(unmoved_data)

        ants.image_write(unmoved, os.path.join(out_dir, base + '_norm.nii.gz'))
        # try to keep memory usage low w/ manual garbage collection
        del img
        gc.collect()
