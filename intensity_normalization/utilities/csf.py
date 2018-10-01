#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.csf

functions to create the CSF control mask
separated from other routines since it relies on ANTsPy

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 08, 2018
"""

from functools import reduce
import logging
from operator import add
import os

import ants
import numpy as np

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import io, mask

logger = logging.getLogger(__name__)


def csf_mask(img, brain_mask, contrast='t1', csf_thresh=0.9, return_prob=False, mrf=0.25, use_fcm=False):
    """
    create a binary mask of csf using atropos (FMM) segmentation
    of a T1-w image

    Args:
        img (ants.core.ants_image.ANTsImage or nibabel.nifti1.Nifti1Image): target img
        brain_mask (ants.core.ants_image.ANTsImage or nibabel.nifti1.Nifti1Image): brain mask for img
        contrast (str): contrast of the img (e.g., t1, t2, or flair)
        csf_thresh (float): membership threshold to count as CSF
        return_prob (bool): if true, then return membership values
            instead of binary (i.e., thresholded membership) mask
        mrf (float): markov random field parameter
            (i.e., smoothness parameter, higher is a smoother segmentation)
        use_fcm (bool): use FCM segmentation instead of atropos (may be less accurate)
            cannot use return_prob flag
    Returns:
        csf (np.ndarray): binary CSF mask for img
    """
    # convert nibabel to antspy format images (to do atropos segmentation)
    if hasattr(img, 'get_data') and hasattr(brain_mask, 'get_data') and not use_fcm:
        img = nibabel_to_ants(img)
        brain_mask = nibabel_to_ants(brain_mask)
    if not use_fcm:
        res = img.kmeans_segmentation(3, kmask=brain_mask, mrf=mrf)
        avg_intensity = [np.mean(img.numpy()[prob_img.numpy() > 0.5]) for prob_img in res['probabilityimages']]
        csf_arg = np.argmin(avg_intensity) if contrast.lower() in ('t1', 'flair') else np.argmax(avg_intensity)
        csf = res['probabilityimages'][csf_arg].numpy()
        if not return_prob:
            csf = (csf > csf_thresh).astype(np.float32)
    else:
        if hasattr(img, 'numpy') and hasattr(brain_mask, 'numpy'):
            img = to_nibabel(img)
            brain_mask = to_nibabel(brain_mask)
        seg = mask.fcm_class_mask(img, brain_mask, hard_seg=True)
        avg_intensity = [np.mean(img.get_data()[seg == i]) for i in range(1, 4)]
        csf_arg = np.argmin(avg_intensity) if contrast.lower() in ('t1', 'flair') else np.argmax(avg_intensity)
        csf = (seg == (csf_arg + 1)).astype(np.float32)
    return csf


def csf_mask_intersection(img_dir, masks=None, prob=1):
    """
    use all nifti T1w images in data_dir to create csf mask in common areas

    Args:
        img_dir (str): directory containing MR images to be normalized
        masks (str or ants.core.ants_image.ANTsImage): if images are not skull-stripped,
            then provide brain mask as either a corresponding directory or an individual mask
        prob (float): given all data, proportion of data labeled as csf to be
            used for intersection

    Returns:
        intersection (np.ndarray): binary mask of common csf areas for all provided imgs
    """
    if not (0 <= prob <= 1):
        raise NormalizationError('prob must be between 0 and 1. {} given.'.format(prob))
    data = io.glob_nii(img_dir)
    masks = io.glob_nii(masks) if isinstance(masks, str) else [masks] * len(data)
    csf = []
    for i, (img, mask) in enumerate(zip(data, masks)):
        _, base, _ = io.split_filename(img)
        logger.info('Creating CSF mask for image {} ({:d}/{:d})'.format(base, i+1, len(data)))
        imgn = ants.image_read(img)
        maskn = ants.image_read(mask) if isinstance(mask, str) else mask
        csf.append(csf_mask(imgn, maskn))
    csf_sum = reduce(add, csf)  # need to use reduce instead of sum b/c data structure
    intersection = np.zeros(csf_sum.shape)
    intersection[csf_sum >= np.floor(len(data) * prob)] = 1
    return intersection


# TODO: remove these functions when ANTsPy releases new binaries, replace use cases with ants.from_nibabel(), ants.to_nibabel()
def nibabel_to_ants(nib_image):
    """ convert a nibabel image to an ants image """
    from tempfile import mktemp
    tmpfile = mktemp(suffix='.nii.gz')
    nib_image.to_filename(tmpfile)
    new_img = ants.image_read(tmpfile)
    os.remove(tmpfile)
    return new_img


def to_nibabel(image):
    """ Convert an ANTsImage to a Nibabel image """
    if image.dimension != 3:
        raise ValueError('Only 3D images currently supported')
    import nibabel as nib
    array_data = image.numpy()
    affine = np.hstack([image.direction*np.diag(image.spacing),np.array(image.origin).reshape(3,1)])
    affine = np.vstack([affine, np.array([0,0,0,1.])])
    new_img = nib.Nifti1Image(array_data, affine)
    return new_img
