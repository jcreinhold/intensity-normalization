#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.lsq

fit the

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 23, 2020
"""

from __future__ import print_function, division

import logging
import os

import nibabel as nib
import numpy as np

from intensity_normalization.utilities import io
from intensity_normalization.utilities import mask as mask_util
from .fcm import fcm_normalize, find_tissue_mask

logger = logging.getLogger(__name__)


def lsq_normalize(img_dir, mask_dir=None, output_dir=None, write_to_disk=True):
    """
    normalize intensities of a set of MR images by minimizing the squared distance
    between CSF, GM, and WM means within the set

    Args:
        img_dir (str): directory containing MR images
        mask_dir (str): directory containing masks for MR images
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        write_to_disk (bool): write the normalized data to disk or nah

    Returns:
        normalized (np.ndarray): last normalized image from img_dir
    """
    input_files = io.glob_nii(img_dir)
    if output_dir is None:
        out_fns = [None] * len(input_files)
    else:
        out_fns = []
        for fn in input_files:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + '_lsq' + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    mask_files = [None] * len(input_files) if mask_dir is None else io.glob_nii(mask_dir)

    standard_tissue_means = None
    normalized = None
    for i, (img_fn, mask_fn, out_fn) in enumerate(zip(input_files, mask_files, out_fns)):
        _, base, _ = io.split_filename(img_fn)
        logger.info('Transforming image {} to standard scale ({:d}/{:d})'.format(base, i+1, len(input_files)))
        img = io.open_nii(img_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        tissue_mem = mask_util.fcm_class_mask(img, mask)
        if standard_tissue_means is None:
            csf_tissue_mask = find_tissue_mask(img, mask, tissue_type='csf')
            csf_normed_data = fcm_normalize(img, csf_tissue_mask).get_fdata()
            standard_tissue_means = calc_tissue_means(csf_normed_data, tissue_mem)
            del csf_tissue_mask, csf_normed_data
        img_data = img.get_fdata()
        tissue_means = calc_tissue_means(img_data, tissue_mem)
        sf = find_scaling_factor(tissue_means, standard_tissue_means)
        logger.debug('Scaling factor for {}: {:0.3e}'.format(base, sf))
        normalized = nib.Nifti1Image(sf * img_data, img.affine, img.header)
        if write_to_disk:
            io.save_nii(normalized, out_fn, is_nii=True)

    return normalized


def calc_tissue_means(img, tissue_mem):
    def wavg(w,x): return (w*x).sum() / w.sum()
    return np.asarray([[wavg(tissue_mem[...,i], img) for i in range(tissue_mem.shape[-1])]]).T


def find_scaling_factor(tissue_means, standard_tissue_means):
    sf = (tissue_means.T @ standard_tissue_means) / (tissue_means.T @ tissue_means)
    return sf.squeeze()
