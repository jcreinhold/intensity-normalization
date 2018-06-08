#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.ravel

Use RAVEL [1] to intensity normalize a population of MR images

Note that this package requires RAVEL (and its dependencies)
to be installed in R

References:
   ﻿[1] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu,
        and R. T. Shinohara, “Removing inter-subject technical variability
        in magnetic resonance imaging studies,” Neuroimage, vol. 132,
        pp. 198–212, 2016.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 27, 2018
"""

from glob import glob
import logging
import os

import nibabel as nib
import numpy as np

from intensity_normalization.errors import NormalizationError
from intensity_normalization.normalize.whitestripe import whitestripe, whitestripe_norm
from intensity_normalization.utilities import io


logger = logging.getLogger(__name__)


def ravel_normalize(img_dir, template_mask, control_mask, contrast,
                    output_dir=None, write_to_disk=False, do_whitestripe=True, k=1):
    """
    Use RAVEL [1] to normalize the intensities of a set of MR images to eliminate
    unwanted technical variation in images (but, hopefully, preserve biological variation)

    Args:
        img_dir (str): directory containing MR images to be normalized
        template_mask (str): brain mask for template image
        control_mask (str): path to mask of control region (e.g., CSF) for data in data_dir
        contrast (str): contrast of MR images to be normalized (T1, T2, or FLAIR)
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        write_to_disk (bool): write the normalized data to disk or nah
        kwargs: ravel keyword arguments not included here

    Returns:
        Z (np.ndarray): unwanted factors (used in ravel correction)
        normalized (np.ndarray): set of normalized images from data_dir

    References:
        [1] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu,
            and R. T. Shinohara, “Removing inter-subject technical variability
            in magnetic resonance imaging studies,” Neuroimage, vol. 132,
            pp. 198–212, 2016.
    """
    img_fns = sorted(glob(os.path.join(img_dir, '*.nii*')))

    if output_dir is None or not write_to_disk:
        out_fns = None
    else:
        out_fns = []
        for fn in img_fns:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    verbose = True if logger.getEffectiveLevel() == logging.getLevelName('DEBUG') else False

    # get parameters necessary and setup the V array
    V, Vc = image_matrix(img_fns, contrast, masks=template_mask,
                         control_mask=control_mask, do_whitestripe=do_whitestripe,
                         verbose=verbose)

    # estimate the unwanted factors Z
    _, _, vh = np.linalg.svd(Vc)
    Z = vh[:, 1:k]

    # perform the ravel correction
    V_norm = ravel_correction(V, Z)

    # save the results to disk if desired
    if write_to_disk:
        for i, (img_fn, out_fn) in enumerate(zip(img_fns, out_fns)):
            img = io.open_nii(img_fn)
            norm = V_norm[:, i].reshape(img.get_data().shape)
            io.save_nii(img, out_fn, data=norm)

    return Z, V_norm


def ravel_correction(V, Z):
    """
    correct the images (in the image matrix V) by removing the trend
    found in Z

    Args:
        V (np.ndarray): image matrix (rows are voxels, columns are images)
        Z (np.ndarray): unwanted factors (see ravel_normalize and the orig paper)

    Returns:
        res (np.ndarray): normalized images
    """
    means = np.mean(V, axis=1)  # row means
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T, Z)), Z.T), V.T)
    fitted = np.matmul(Z, beta).T
    res = V - fitted
    res = res + means[:,np.newaxis]
    return res


def image_matrix(imgs, contrast, masks=None, control_mask=None,
                 do_whitestripe=True, verbose=False):
    """
    creates an matrix of images where the rows correspond the the voxels of
    each image and the columns are the images

    if a control mask is supplied, then a similarly shaped matrix is also output
    where the rows correspond to voxels defined in the control mask

    Args:
        imgs (list):
        contrast (str):
        masks (list or str):
        control_mask (str):
        do_whitestripe (bool):
        verbose (bool):

    Returns:
        V (np.ndarray):
        Vc (np.ndarray):
    """
    img_shape = io.open_nii(imgs[0]).get_data().shape
    V = np.zeros((int(np.prod(img_shape)), len(imgs)))
    if control_mask is not None and masks is not None and isinstance(masks, str):
        cmask = io.open_nii(control_mask)
        mask_ = io.open_nii(masks)
        cmask_data = cmask.get_data() * mask_.get_data()  # make sure that the template and control overlap
        num_c_pts = int(np.sum(cmask_data.flatten()))
        Vc = np.zeros((num_c_pts, len(imgs)))
        masks = [masks] * len(imgs)
    elif control_mask is not None and (masks is None or not isinstance(masks, str)):
        raise NormalizationError('If control mask provided, then *one* template brain mask must be provided')

    if masks is None:
        masks = [None] * len(imgs)

    # do whitestripe on the image before applying RAVEL (if desired)
    for i, (img_fn, mask_fn) in enumerate(zip(imgs, masks)):
        img = io.open_nii(img_fn)
        mask = io.open_nii(mask_fn)
        if do_whitestripe:
            logger.info('Applying WhiteStripe to image {} ({:d}/{:d})'.format(img_fn, i + 1, len(imgs)))
            inds = whitestripe(img, contrast, mask, verbose=verbose)
            img = whitestripe_norm(img, inds)
        img_data = img.get_data()
        V[:,i] = img_data.flatten()
        if control_mask is not None:
            Vc[:,i] = img_data[cmask_data == 1].flatten()

    return V if control_mask is None else (V, Vc)


def image_matrix_to_images(V, imgs):
    """
    convert an image matrix to a list of the correctly formated nifti images

    Args:
        V (np.ndarray):
        imgs (list):

    Returns:
        img_list (list):
    """
    img_list = []
    for i, img_fn in enumerate(imgs):
        img = io.open_nii(img_fn)
        nimg = nib.Nifti1Image(V[:, i].reshape(img.get_data().shape), img.affine, img.header)
        img_list.append(nimg)
    return img_list



