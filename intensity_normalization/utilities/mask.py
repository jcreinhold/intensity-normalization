#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mask


Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 01, 2018
"""

import numpy as np
from skfuzzy import cmeans


def class_mask(img, brain_mask=None, hard_seg=False):
    """
    creates a mask of a target brain

    Args:
        img:
        brain_mask:

    Returns:

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
