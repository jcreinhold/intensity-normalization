# -*- coding: utf-8 -*-
"""
intensity_normalization.util.tissue_membership

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = []

import numpy as np
from skfuzzy import cmeans

from intensity_normalization.type import Array


def find_tissue_memberships(
    image: Array, brain_mask: Array = None, hard_segmentation: bool = False
) -> Array:
    """Tissue memberships for a T1-w brain image with fuzzy c-means

    Args:
        image: image to find tissue masks for (must be T1-w)
        brain_mask: mask covering the brain of image (none if already skull-stripped)
        hard_segmentation: pick the maximum membership as the true class in output

    Returns:
        tissue_mask: membership values for each of three classes in the image
            (or class determinations w/ hard_seg)
    """
    if brain_mask is None:
        brain_mask = image > 0.0
    else:
        brain_mask = brain_mask > 0.0
    mask_size = brain_mask.sum()
    foreground = image[brain_mask]
    t1_cntr, t1_mem, _, _, _, _, _ = cmeans(
        foreground.reshape(-1, mask_size), 3, 2, 0.005, 50
    )
    t1_mem_list = [
        t1_mem[i] for i, _ in sorted(enumerate(t1_cntr), key=lambda x: x[1])
    ]  # sort the tissue memberships to CSF/GM/WM
    tissue_mask = np.zeros(image.shape + (3,))
    for i in range(3):
        tissue_mask[..., i][brain_mask] = t1_mem_list[i]
    if hard_segmentation:
        tmp_mask = np.zeros(image.shape)
        masked = tissue_mask[brain_mask]
        tmp_mask[brain_mask] = np.argmax(masked, axis=1) + 1
        tissue_mask = tmp_mask
    return tissue_mask
