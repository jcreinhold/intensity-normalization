#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
csf


Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 08, 2018
"""

from functools import reduce
from glob import glob
import logging
from operator import add
import os

import ants
import numpy as np

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import io

logger = logging.getLogger(__name__)


def csf_mask(img, brain_mask, csf_thresh=0.9):
    """
    create a binary mask of csf using atropos (FMM) segmentation
    of a T1-w image

    Args:
        img (ants.core.ants_image.ANTsImage): target img
        brain_mask (ants.core.ants_image.ANTsImage): brain mask for img
        csf_thresh (float): membership threshold to count as CSF

    Returns:
        csf (np.ndarray): binary CSF mask for img
    """
    res = img.kmeans_segmentation(3, kmask=brain_mask, mrf=0.3)
    avg_intensity = [np.mean(img.numpy()[prob_img.numpy() > csf_thresh]) for prob_img in res['probabilityimages']]
    csf_arg = np.argmin(avg_intensity)
    csf = (res['probabilityimages'][csf_arg].numpy() > csf_thresh).astype(np.float32)
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
    data = sorted(glob(os.path.join(img_dir, '*.nii*')))
    masks = sorted(glob(os.path.join(masks, '*.nii*'))) if isinstance(masks, str) else [masks] * len(data)
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
