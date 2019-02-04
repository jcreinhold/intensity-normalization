#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.mask

create a tissue class mask of a target image

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 01, 2018
"""

from __future__ import print_function, division

import logging
import warnings

import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import (binary_closing, binary_fill_holes, generate_binary_structure, iterate_structure,
                                      binary_dilation)
from skfuzzy import cmeans
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from intensity_normalization.errors import NormalizationError

logger = logging.getLogger(__name__)


def fcm_class_mask(img, brain_mask=None, hard_seg=False):
    """
    creates a mask of tissue classes for a target brain with fuzzy c-means

    Args:
        img (nibabel.nifti1.Nifti1Image): target image (must be T1w)
        brain_mask (nibabel.nifti1.Nifti1Image): mask covering the brain of img
            (none if already skull-stripped)
        hard_seg (bool): pick the maximum membership as the true class in output

    Returns:
        mask (np.ndarray): membership values for each of three classes in the image
            (or class determinations w/ hard_seg)
    """
    img_data = img.get_data()
    if brain_mask is not None:
        mask_data = brain_mask.get_data() > 0
    else:
        mask_data = img_data > img_data.mean()
    [t1_cntr, t1_mem, _, _, _, _, _] = cmeans(img_data[mask_data].reshape(-1, len(mask_data[mask_data])),
                                              3, 2, 0.005, 50)
    t1_mem_list = [t1_mem[i] for i, _ in sorted(enumerate(t1_cntr), key=lambda x: x[1])]  # CSF/GM/WM
    mask = np.zeros(img_data.shape + (3,))
    for i in range(3):
        mask[..., i][mask_data] = t1_mem_list[i]
    if hard_seg:
        tmp_mask = np.zeros(img_data.shape)
        tmp_mask[mask_data] = np.argmax(mask[mask_data], axis=1) + 1
        mask = tmp_mask
    return mask


def gmm_class_mask(img, brain_mask=None, contrast='t1', return_wm_peak=True, hard_seg=False):
    """
    get a tissue class mask using gmms (or just the WM peak, for legacy use)

    Args:
        img (nibabel.nifti1.Nifti1Image): target img
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
            (none if already skull-stripped)
        contrast (str): string to describe img's MR contrast
        return_wm_peak (bool): if true, return only the wm peak
        hard_seg (bool): if true and return_wm_peak false, then return
            hard segmentation of tissue classes

    Returns:
        if return_wm_peak true:
            wm_peak (float): represents the mean intensity for WM
        else:
            mask (np.ndarray):
                if hard_seg, then mask is the same size as img
                else, mask is the same size as img * 3, where
                the new dimensions hold the probabilities of tissue class
    """
    img_data = img.get_data()
    if brain_mask is not None:
        mask_data = brain_mask.get_data() > 0
    else:
        mask_data = img_data > img_data.mean()

    brain = np.expand_dims(img_data[mask_data].flatten(), 1)
    gmm = GaussianMixture(3)
    gmm.fit(brain)

    if return_wm_peak:
        means = sorted(gmm.means_.T.squeeze())
        if contrast.lower() == 't1':
            wm_peak = means[2]
        elif contrast.lower() == 'flair':
            wm_peak = means[1]
        elif contrast.lower() == 't2':
            wm_peak = means[0]
        else:
            raise NormalizationError('Invalid contrast type: {}. Must be `t1`, `t2`, or `flair`.'.format(contrast))
        return wm_peak
    else:
        classes_ = np.argsort(gmm.means_.T.squeeze())
        if contrast.lower() == 't1':
            classes = [classes_[0], classes_[1], classes_[2]]
        elif contrast.lower() == 'flair':
            classes = [classes_[0], classes_[2], classes_[1]]
        elif contrast.lower() == 't2':
            classes = [classes_[2], classes_[1], classes_[0]]
        else:
            raise NormalizationError('Invalid contrast type: {}. Must be `t1`, `t2`, or `flair`.'.format(contrast))
        if hard_seg:
            tmp_predicted = gmm.predict(brain)
            predicted = np.zeros(tmp_predicted.shape)
            for i, c in enumerate(classes):
                predicted[tmp_predicted == c] = i + 1
            mask = np.zeros(img_data.shape)
            mask[mask_data] = predicted + 1
        else:
            predicted_proba = gmm.predict_proba(brain)
            mask = np.zeros((*img_data.shape, 3))
            for i, c in enumerate(classes):
                mask[mask_data, i] = predicted_proba[:, c]
        return mask


def __fill_2p5d(img):
    """ helper function for background_mask """
    out_img = np.zeros_like(img)
    for slice_num in range(img.shape[2]):
        out_img[:, :, slice_num] = binary_fill_holes(img[:, :, slice_num])
    return out_img


def background_mask(img, seed=0):
    """
    create a background mask for a given mr img

    Args:
        img (nibabel.nifti1.Nifti1Image): img from which to extract background
        seed (int): since random sampling used, pick seed for reproducibility

    Returns:
        background (nibabel.nifti1.Nifti1Image): background mask
    """
    np.random.seed(seed)
    logger.info('Finding Background...')
    img_data = img.get_data()
    km = KMeans(4)
    rand_mask = np.random.rand(*img_data.shape) > 0.75
    logger.info('Fitting KMeans...')
    km.fit(np.expand_dims(img_data[rand_mask], 1))
    logger.info('Generating Mask...')
    classes = km.predict(np.expand_dims(img_data.flatten(), 1)).reshape(img_data.shape)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        means = [np.mean(img_data[classes == i]) for i in range(4)]
    raw_mask = (classes == np.argmin(means)) == 0.0
    filled_raw_mask = __fill_2p5d(raw_mask)
    dist2_5by5_kernel = iterate_structure(generate_binary_structure(3, 1), 2)
    closed_mask = binary_closing(filled_raw_mask, dist2_5by5_kernel, 5)
    filled_closed_mask = __fill_2p5d(np.logical_or(closed_mask, filled_raw_mask)).astype(np.float32)
    bg_mask = binary_dilation(filled_closed_mask, generate_binary_structure(3, 1), 2)
    background = nib.Nifti1Image(bg_mask, img.affine, img.header)
    return background
