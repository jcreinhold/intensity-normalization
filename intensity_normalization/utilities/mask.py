#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.mask

create a tissue class mask of a target image

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 01, 2018
"""

import numpy as np
from skfuzzy import cmeans
from sklearn.mixture import GaussianMixture


def fcm_class_mask(img, brain_mask=None, hard_seg=False):
    """
    creates a mask of tissue classes for a target brain with fuzzy c-means

    Args:
        img: target image nifti object
        brain_mask: mask nifti object that covers the brain of the img
        hard_seg (bool): pick the maximum membership as the true class in output

    Returns:
        mask (np.ndarray): membership values for each of three classes in the image
            (or class determinations w/ hard_seg)
    """
    img_data = img.get_data()
    if brain_mask is not None:
        mask_data = brain_mask.get_data() > 0
    else:
        mask_data = img_data > 0
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


def gmm_class_mask(img, brain_mask=None, contrast='t1'):
    img_data = img.get_data()
    if brain_mask is not None:
        mask_data = brain_mask.get_data() > 0
    else:
        mask_data = img_data > 0

    gmm = GaussianMixture(3)
    gmm.fit(np.expand_dims(img_data[mask_data == 1].flatten(), 1))

    means = gmm.means_.T.tolist()[0]
    weights = gmm.weights_.tolist()

    wm_peak = max(means) if contrast == 't1' else \
            max(zip(means, weights), key=lambda x: x[1])[0]
    return wm_peak
