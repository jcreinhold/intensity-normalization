#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.ravel

Use RAVEL [1] to intensity normalize a population of MR images

References:
   ﻿[1] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu,
        and R. T. Shinohara, “Removing inter-subject technical variability
        in magnetic resonance imaging studies,” NeuroImage, vol. 132,
        pp. 198–212, 2016.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 27, 2018
"""

from functools import reduce
import gc
import logging
from operator import add
import os

import ants
import nibabel as nib
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import svds

from intensity_normalization.errors import NormalizationError
from intensity_normalization.normalize.whitestripe import whitestripe, whitestripe_norm
from intensity_normalization.utilities import csf
from intensity_normalization.utilities import io

logger = logging.getLogger(__name__)


def ravel_normalize(
    image_dir,
    mask_dir,
    contrast,
    output_dir=None,
    write_to_disk=False,
    do_whitestripe=True,
    b=1,
    membership_thresh=0.99,
    segmentation_smoothness=0.25,
    do_registration=False,
    use_fcm=True,
    sparse_svd=False,
    csf_masks=False,
):
    """
    Use RAVEL [1] to normalize the intensities of a set of MR images to eliminate
    unwanted technical variation in images (but, hopefully, preserve biological variation)

    this function has an option that is modified from [1] in where no registration is done,
    the control mask is defined dynamically by finding a tissue segmentation of the brain and
    thresholding the membership at a very high level (this seems to work well and is *much* faster)
    but there seems to be some more inconsistency in the results

    Args:
        image_dir (str): directory containing MR images to be normalized
        mask_dir (str): brain masks for imgs (or csf masks if csf_masks is True)
        contrast (str): contrast of MR images to be normalized (T1, T2, or FLAIR)
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        write_to_disk (bool): write the normalized data to disk or nah
        do_whitestripe (bool): whitestripe normalize the images before applying RAVEL correction
        b (int): number of unwanted factors to estimate
        membership_thresh (float): threshold of membership for control voxels
        segmentation_smoothness (float): segmentation smoothness parameter for atropos ANTsPy
            segmentation scheme (i.e., mrf parameter)
        do_registration (bool): deformably register images to find control mask
        use_fcm (bool): use FCM for segmentation instead of atropos (may be less accurate)
        sparse_svd (bool): use traditional SVD (LAPACK) to calculate right singular vectors
            else use ARPACK
        csf_masks (bool): provided masks are the control masks (not brain masks)
            assumes that images are deformably co-registered

    Returns:
        Z (np.ndarray): unwanted factors (used in ravel correction)
        normalized (np.ndarray): set of normalized images from data_dir

    References:
        [1] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu,
            and R. T. Shinohara, “Removing inter-subject technical variability
            in magnetic resonance imaging studies,” Neuroimage, vol. 132,
            pp. 198–212, 2016.
    """
    img_fns = io.glob_nii(image_dir)
    mask_fns = io.glob_nii(mask_dir)

    if output_dir is None or not write_to_disk:
        out_fns = None
    else:
        out_fns = []
        for fn in img_fns:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # get parameters necessary and setup the V array
    V, Vc = image_matrix(
        img_fns,
        contrast,
        masks=mask_fns,
        do_whitestripe=do_whitestripe,
        return_ctrl_matrix=True,
        membership_thresh=membership_thresh,
        do_registration=do_registration,
        smoothness=segmentation_smoothness,
        use_fcm=use_fcm,
        csf_masks=csf_masks,
    )

    # estimate the unwanted factors Z
    _, _, vh = (
        np.linalg.svd(Vc, full_matrices=False)
        if not sparse_svd
        else svds(bsr_matrix(Vc), k=b, return_singular_vectors="vh")
    )
    Z = vh.T[:, 0:b]

    # perform the ravel correction
    V_norm = ravel_correction(V, Z)

    # save the results to disk if desired
    if write_to_disk:
        for i, (img_fn, out_fn) in enumerate(zip(img_fns, out_fns)):
            img = io.open_nii(img_fn)
            norm = V_norm[:, i].reshape(img.get_fdata().shape)
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
    fitted = np.matmul(
        Z, beta
    ).T  # this line (alone) gives slightly diff answer than R ver, otherwise exactly same
    res = V - fitted
    res = res + means[:, np.newaxis]
    return res


def image_matrix(
    images,
    modality,
    masks=None,
    do_whitestripe=True,
    return_ctrl_matrix=False,
    membership_thresh=0.99,
    smoothness=0.25,
    max_ctrl_vox=10000,
    do_registration=False,
    ctrl_prob=1,
    use_fcm=False,
    csf_masks=False,
):
    """
    creates an matrix of images where the rows correspond the the voxels of
    each image and the columns are the images

    Args:
        images (list): list of paths to MR images of interest
        modality (str): contrast of the set of imgs (e.g., T1)
        masks (list or str): list of corresponding brain masks or just one (template) mask
        do_whitestripe (bool): do whitestripe on the images before storing in matrix or nah
        return_ctrl_matrix (bool): return control matrix for imgs (i.e., a subset of V's rows)
        membership_thresh (float): threshold of membership for control voxels (want this very high)
            this option is only used if the registration is turned off
        smoothness (float): smoothness parameter for segmentation for control voxels
            this option is only used if the registration is turned off
        max_ctrl_vox (int): maximum number of control voxels (if too high, everything
            crashes depending on available memory) only used if do_registration is false
        do_registration (bool): register the images together and take the intersection of the csf
            masks (as done in the original paper, note that this takes much longer)
        ctrl_prob (float): given all data, proportion of data labeled as csf to be
            used for intersection (i.e., when do_registration is true)
        use_fcm (bool): use FCM for segmentation instead of atropos (may be less accurate)
        csf_masks (bool): provided masks are the control masks (not brain masks)
            assumes that images are deformably co-registered

    Returns:
        V (np.ndarray): image matrix (rows are voxels, columns are images)
        Vc (np.ndarray): image matrix of control voxels (rows are voxels, columns are images)
            Vc only returned if return_ctrl_matrix is True
    """
    img_shape = io.open_nii(images[0]).get_fdata().shape
    V = np.zeros((int(np.prod(img_shape)), len(images)))

    if return_ctrl_matrix:
        ctrl_vox = []

    if masks is None and return_ctrl_matrix:
        raise NormalizationError(
            "Brain masks must be provided if returning control memberships"
        )
    if masks is None:
        masks = [None] * len(images)

    do_registration = do_registration and not csf_masks

    for i, (image_fn, mask_fn) in enumerate(zip(images, masks)):
        _, base, _ = io.split_filename(image_fn)
        image = io.open_nii(image_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        # do whitestripe on the image before applying RAVEL (if desired)
        if do_whitestripe:
            logger.info(
                "Applying WhiteStripe to image {} ({:d}/{:d})".format(
                    base, i + 1, len(images)
                )
            )
            inds = whitestripe(image, modality, mask)
            image = whitestripe_norm(image, inds)
        data = image.get_fdata()
        if data.shape != img_shape:
            raise NormalizationError(
                "Cannot normalize because image {} needs to have same dimension "
                "as all other images ({} != {})".format(base, data.shape, img_shape)
            )
        V[:, i] = data.flatten()
        if return_ctrl_matrix:
            if do_registration and i == 0:
                logger.info(
                    "Creating control mask for image {} ({:d}/{:d})".format(
                        base, i + 1, len(images)
                    )
                )
                verbose = (
                    True
                    if logger.getEffectiveLevel() == logging.getLevelName("DEBUG")
                    else False
                )
                ctrl_masks = []
                reg_imgs = []
                reg_imgs.append(csf.nibabel_to_ants(image))
                ctrl_masks.append(
                    csf.csf_mask(
                        image,
                        mask,
                        contrast=modality,
                        csf_thresh=membership_thresh,
                        mrf=smoothness,
                        use_fcm=use_fcm,
                    )
                )
            elif do_registration and i != 0:
                template = ants.image_read(images[0])
                tmask = ants.image_read(masks[0])
                image = csf.nibabel_to_ants(image)
                logger.info(
                    "Starting registration for image {} ({:d}/{:d})".format(
                        base, i + 1, len(images)
                    )
                )
                reg_result = ants.registration(
                    template,
                    image,
                    type_of_transform="SyN",
                    mask=tmask,
                    verbose=verbose,  # noqa
                )
                image = reg_result["warpedmovout"]
                mask = csf.nibabel_to_ants(mask)
                reg_imgs.append(image)  # noqa
                logger.info(
                    "Creating control mask for image {} ({:d}/{:d})".format(
                        base, i + 1, len(images)
                    )
                )
                ctrl_masks.append(  # noqa
                    csf.csf_mask(
                        image,
                        mask,
                        contrast=modality,
                        csf_thresh=membership_thresh,
                        mrf=smoothness,
                        use_fcm=use_fcm,
                    )
                )
            else:  # assume pre-registered
                logger.info(
                    "Finding control voxels for image {} ({:d}/{:d})".format(
                        base, i + 1, len(images)
                    )
                )
                ctrl_mask = (
                    csf.csf_mask(
                        image,
                        mask,
                        contrast=modality,
                        csf_thresh=membership_thresh,
                        mrf=smoothness,
                        use_fcm=use_fcm,
                    )
                    if csf_masks
                    else mask.get_fdata()
                )
                if np.sum(ctrl_mask) == 0:
                    raise NormalizationError(
                        "No control voxels found for image ({}) at threshold ({})".format(
                            base, membership_thresh
                        )
                    )
                elif np.sum(ctrl_mask) < 100:
                    logger.warning(
                        "Few control voxels found ({:d}) (potentially a problematic image ({}) or "
                        "threshold ({}) too high)".format(
                            int(np.sum(ctrl_mask)), base, membership_thresh
                        )
                    )
                ctrl_vox.append(data[ctrl_mask == 1].flatten())  # noqa

    if return_ctrl_matrix and not do_registration:
        min_len = min(min(map(len, ctrl_vox)), max_ctrl_vox)
        logger.info("Using {:d} control voxels".format(min_len))
        Vc = np.zeros((min_len, len(images)))
        for i in range(len(images)):
            ctrl_voxs = ctrl_vox[i][:min_len]
            logger.info(
                "Image {:d} control voxel stats -  mean: {:.3f}, std: {:.3f}".format(
                    i + 1, np.mean(ctrl_voxs), np.std(ctrl_voxs)
                )
            )
            Vc[:, i] = ctrl_voxs
    elif return_ctrl_matrix and do_registration:
        ctrl_sum = reduce(
            add, ctrl_masks
        )  # need to use reduce instead of sum b/c data structure
        intersection = np.zeros(ctrl_sum.shape)
        intersection[ctrl_sum >= np.floor(len(ctrl_masks) * ctrl_prob)] = 1
        num_ctrl_vox = int(np.sum(intersection))
        Vc = np.zeros((num_ctrl_vox, len(images)))
        for i, image in enumerate(reg_imgs):
            ctrl_voxs = image.numpy()[intersection == 1]
            logger.info(
                "Image {:d} control voxel stats -  mean: {:.3f}, std: {:.3f}".format(
                    i + 1, np.mean(ctrl_voxs), np.std(ctrl_voxs)
                )
            )
            Vc[:, i] = ctrl_voxs
        del ctrl_masks, reg_imgs
        gc.collect()  # force a garbage collection, since we just used the majority of the system memory

    return V if not return_ctrl_matrix else (V, Vc)  # noqa


def image_matrix_to_images(V, images):
    """
    convert an image matrix to a list of the correctly formated nifti images

    Args:
        V (np.ndarray): image matrix (rows are voxels, columns are images)
        images (list): list of paths to corresponding MR images in V

    Returns:
        img_list (list): list of nifti images extracted from V
    """
    img_list = []
    for i, image_fn in enumerate(images):
        image = io.open_nii(image_fn)
        nimg = nib.Nifti1Image(
            V[:, i].reshape(image.get_fdata().shape), image.affine, image.header
        )
        img_list.append(nimg)
    return img_list
