#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.ravel

Use RAVEL [1] to intensity normalize a population of MR images

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
from intensity_normalization.utilities import csf
from intensity_normalization.utilities import io

logger = logging.getLogger(__name__)


def ravel_normalize(img_dir, mask_dir, contrast, output_dir=None, write_to_disk=False,
                    do_whitestripe=True, b=1, membership_thresh=0.99):
    """
    Use RAVEL [1] to normalize the intensities of a set of MR images to eliminate
    unwanted technical variation in images (but, hopefully, preserve biological variation)

    this is modified from [1] in that *no* registration is done, the control mask is defined
    dynamically by finding a tissue segmentation of the brain and thresholding the membership
    at a very high level (this seems to work well and is *much* faster)

    Args:
        img_dir (str): directory containing MR images to be normalized
        mask_dir (str): brain masks for imgs
        contrast (str): contrast of MR images to be normalized (T1, T2, or FLAIR)
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        write_to_disk (bool): write the normalized data to disk or nah
        do_whitestripe (bool): whitestripe normalize the images before applying RAVEL correction
        b (int): number of unwanted factors to estimate
        membership_thresh (float): threshold of membership for control voxels

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
    mask_fns = sorted(glob(os.path.join(mask_dir, '*.nii*')))

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
    V, Vc = image_matrix(img_fns, contrast, masks=mask_fns, do_whitestripe=do_whitestripe,
                         verbose=verbose, return_ctrl_matrix=True, membership_thresh=membership_thresh)

    # estimate the unwanted factors Z
    _, _, vh = np.linalg.svd(Vc)
    Z = vh.T[:, 0:b]

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
        Z (np.ndarray): unwanted factors (see ravel_normalize.py and the orig paper)

    Returns:
        res (np.ndarray): normalized images
    """
    means = np.mean(V, axis=1)  # row means
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T, Z)), Z.T), V.T)
    fitted = np.matmul(Z, beta).T  # this line (alone) gives slightly diff answer than R ver, otherwise exactly same
    res = V - fitted
    res = res + means[:,np.newaxis]
    return res


def image_matrix(imgs, contrast, masks=None, do_whitestripe=True, return_ctrl_matrix=False,
                 membership_thresh=0.99, smoothness=0.25, max_ctrl_vox=10000, verbose=False):
    """
    creates an matrix of images where the rows correspond the the voxels of
    each image and the columns are the images

    Args:
        imgs (list): list of paths to MR images of interest
        contrast (str): contrast of the set of imgs (e.g., T1)
        masks (list or str): list of corresponding brain masks or just one (template) mask
        do_whitestripe (bool): do whitestripe on the images before storing in matrix or nah
        return_ctrl_matrix (bool): return control matrix for imgs (i.e., a subset of V's rows)
        membership_thresh (float): threshold of membership for control voxels (want this very high)
        smoothness (float): smoothness parameter for segmentation for control voxels
        max_ctrl_vox (int): maximum number of control voxels (if too high, everything
            crashes depending on available memory)
        verbose (bool): pass verbosity option to whitestripe if desired

    Returns:
        V (np.ndarray): image matrix (rows are voxels, columns are images)
        Vc (np.ndarray): image matrix of control voxels (rows are voxels, columns are images)
            Vc only returned if return_ctrl_matrix is True
    """
    img_shape = io.open_nii(imgs[0]).get_data().shape
    V = np.zeros((int(np.prod(img_shape)), len(imgs)))

    if return_ctrl_matrix:
        ctrl_vox = []

    if masks is None and return_ctrl_matrix:
        raise NormalizationError('Brain masks must be provided if returning control memberships')
    if masks is None:
        masks = [None] * len(imgs)

    # do whitestripe on the image before applying RAVEL (if desired)
    for i, (img_fn, mask_fn) in enumerate(zip(imgs, masks)):
        _, base, _ = io.split_filename(img_fn)
        img = io.open_nii(img_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        if do_whitestripe:
            logger.info('Applying WhiteStripe to image {} ({:d}/{:d})'.format(base, i + 1, len(imgs)))
            inds = whitestripe(img, contrast, mask, verbose=verbose)
            img = whitestripe_norm(img, inds)
        img_data = img.get_data()
        V[:,i] = img_data.flatten()
        if return_ctrl_matrix:
            logger.info('Finding control voxels for image {} ({:d}/{:d})'.format(base, i + 1, len(imgs)))
            ctrl_mask = csf.csf_mask(img, mask, csf_thresh=membership_thresh, mrf=smoothness)
            if np.sum(ctrl_mask) == 0:
                raise NormalizationError('No control voxels found for image ({}) at threshold ({})'
                                         .format(base, membership_thresh))
            elif np.sum(ctrl_mask) < 100:
                logger.warning('Few control voxels found ({:d}) (potentially a problematic image ({}) or '
                               'threshold ({}) too high)'.format(int(np.sum(ctrl_mask)), base, membership_thresh))
            ctrl_vox.append(img_data[ctrl_mask == 1].flatten())

    if return_ctrl_matrix:
        min_len = min(min(map(len, ctrl_vox)), max_ctrl_vox)
        logger.info('Using {:d} control voxels'.format(min_len))
        Vc = np.zeros((min_len, len(imgs)))
        for i in range(len(imgs)):
            ctrl_voxs = ctrl_vox[i][:min_len]
            logger.info('Image {:d} control voxel stats -  mean: {:.3f}, std: {:.3f}'
                         .format(i+1, np.mean(ctrl_voxs), np.std(ctrl_voxs)))
            Vc[:,i] = ctrl_voxs

    return V if not return_ctrl_matrix else (V, Vc)


def image_matrix_to_images(V, imgs):
    """
    convert an image matrix to a list of the correctly formated nifti images

    Args:
        V (np.ndarray): image matrix (rows are voxels, columns are images)
        imgs (list): list of paths to corresponding MR images in V

    Returns:
        img_list (list): list of nifti images extracted from V
    """
    img_list = []
    for i, img_fn in enumerate(imgs):
        img = io.open_nii(img_fn)
        nimg = nib.Nifti1Image(V[:, i].reshape(img.get_data().shape), img.affine, img.header)
        img_list.append(nimg)
    return img_list
