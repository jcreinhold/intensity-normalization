#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalize.utilities.register

handles required registration for intensity normalization routines

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 08, 2018
"""

from glob import glob
import logging
import os

import ants
import numpy as np

from intensity_normalization.utilities.io import split_filename

logger = logging.getLogger(__name__)


def register_to_template(img_dir, out_dir=None, tx_dir=None, template_img=0):
    """
    register a set of images using SyN (deformable registration)
    and write their output and transformations to disk

    Args:
        img_dir (str): directory containing MR images registered to a template
        out_dir (str): directory to save images if you do not want them saved in
            a newly created directory (or existing dir) called `normalize_reg`
        tx_dir (str): directory to save registration tforms if you do not want them saved in
            a newly created directory (or existing dir) called `normalize_reg_tforms`
        template_img (int or str): number of img in img_dir, or a specified img path
            to be used as the template which all images are registered to

    Returns:
        None, writes registration transforms and registered images to disk
    """

    img_fns = glob(os.path.join(img_dir, '*.nii*'))

    if isinstance(template_img, int):
        template_img = img_fns[template_img]

    if tx_dir is None:
        tx_dir = os.path.join(os.getcwd(), 'reg_tforms')
        if os.path.exists(tx_dir):
            logger.warning('normalize_reg_tforms directory already exists,'
                           'may overwrite existing tforms!')

    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'registered')
        if os.path.exists(out_dir):
            logger.warning('normalize_reg directory already exists,'
                           'may overwrite existing registered images!')

    img_fns = list(set(img_fns) - set(template_img))
    template = ants.image_read(template_img)

    for fn in img_fns:
        img = ants.image_read(fn)
        _, base, _ = split_filename(fn)
        tx = ants.registration(template, img, type_of_transform='SyN')
        moved = ants.apply_transforms(fixed=template, moving=img, interpolator='linear',
                                      transformlist=tx['fwdtransforms'])
        np.save(os.path.join(tx_dir, base + '_tx.npy'), tx)
        ants.image_write(moved, os.path.join(out_dir, base + '_reg.nii.gz'))


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
    tx_fns = glob(os.path.join(tx_dir, '*.npy'))
    template = ants.image_read(template_img)
    if out_dir is None:
        tx_dir = os.path.join(os.getcwd(), 'normalized')
        if os.path.exists(tx_dir):
            logger.warning('normalize_unreg directory already exists,'
                           'may overwrite existing registered images!')
    for fn, tx_fn in zip(reg_fns, tx_fns):
        _, base, _ = split_filename(fn)
        img = ants.image_read(fn)
        tx = np.load(tx_fn)
        unmoved = ants.apply_transforms(fixed=template, moving=img, interpolator='linear',
                                        transformlist=tx['fwdtransforms'])
        ants.image_write(unmoved,os.path.join(out_dir, base + '_norm.nii.gz'))
