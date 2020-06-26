#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.nyul

Use the method of Nyul and Udupa [1] (updated in [2])
to do piecewise affine histogram-based intensity normalization on a
population of MR images

References:
    [1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image
        Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081,
        1999.
    [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold,
        D. L. Collins, and T. Arbel, “Evaluating intensity
        normalization on MRIs of human brain with multiple sclerosis,”
        Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 01, 2018
"""

from __future__ import print_function, division

import logging
import os

import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d

from intensity_normalization.utilities import io

logger = logging.getLogger(__name__)


def nyul_normalize(img_dir, mask_dir=None, output_dir=None, standard_hist=None, write_to_disk=True):
    """
    Use Nyul and Udupa method ([1,2]) to normalize the intensities of a set of MR images

    Args:
        img_dir (str): directory containing MR images
        mask_dir (str): directory containing masks for MR images
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        standard_hist (str): path to output or use standard histogram landmarks
        write_to_disk (bool): write the normalized data to disk or nah

    Returns:
        normalized (np.ndarray): last normalized image from img_dir

    References:
        [1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image
            Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081,
            1999.
        [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold,
            D. L. Collins, and T. Arbel, “Evaluating intensity
            normalization on MRIs of human brain with multiple sclerosis,”
            Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.
    """
    input_files = io.glob_nii(img_dir)
    if output_dir is None:
        out_fns = [None] * len(input_files)
    else:
        out_fns = []
        for fn in input_files:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + '_hm' + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    mask_files = [None] * len(input_files) if mask_dir is None else io.glob_nii(mask_dir)

    if standard_hist is None:
        logger.info('Learning standard scale for the set of images')
        standard_scale, percs = train(input_files, mask_files)
    elif not os.path.isfile(standard_hist):
        logger.info('Learning standard scale for the set of images')
        standard_scale, percs = train(input_files, mask_files)
        np.save(standard_hist, np.vstack((standard_scale, percs)))
    else:
        logger.info('Loading standard scale ({}) for the set of images'.format(standard_hist))
        standard_scale, percs = np.load(standard_hist)

    normalized = None
    for i, (img_fn, mask_fn, out_fn) in enumerate(zip(input_files, mask_files, out_fns)):
        _, base, _ = io.split_filename(img_fn)
        logger.info('Transforming image {} to standard scale ({:d}/{:d})'.format(base, i+1, len(input_files)))
        img = io.open_nii(img_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        normalized = do_hist_norm(img, percs, standard_scale, mask)
        if write_to_disk:
            io.save_nii(normalized, out_fn, is_nii=True)

    return normalized


def get_landmarks(img, percs):
    """
    get the landmarks for the Nyul and Udupa norm method for a specific image

    Args:
        img (np.ndarray): image on which to find landmarks
        percs (np.ndarray): corresponding landmark percentiles to extract

    Returns:
        landmarks (np.ndarray): intensity values corresponding to percs in img
    """
    landmarks = np.percentile(img, percs)
    return landmarks


def train(img_fns, mask_fns=None, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """
    determine the standard scale for the set of images

    Args:
        img_fns (list): set of NifTI MR image paths which are to be normalized
        mask_fns (list): set of corresponding masks (if not provided, estimated)
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)

    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    mask_fns = [None] * len(img_fns) if mask_fns is None else mask_fns
    percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile+1, step), [i_max]))
    standard_scale = np.zeros(len(percs))
    for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns)):
        img_data = io.open_nii(img_fn).get_data()
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        mask_data = img_data > img_data.mean() if mask is None else mask.get_data()
        masked = img_data[mask_data > 0]
        landmarks = get_landmarks(masked, percs)
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])
        landmarks = np.array(f(landmarks))
        standard_scale += landmarks
    standard_scale = standard_scale / len(img_fns)
    return standard_scale, percs


def do_hist_norm(img, landmark_percs, standard_scale, mask=None):
    """
    do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks

    Args:
        img (nibabel.nifti1.Nifti1Image): image on which to find landmarks
        landmark_percs (np.ndarray): corresponding landmark points of standard scale
        standard_scale (np.ndarray): landmarks on the standard scale
        mask (nibabel.nifti1.Nifti1Image): foreground mask for img

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): normalized image
    """
    img_data = img.get_data()
    mask_data = img_data > img_data.mean() if mask is None else mask.get_data()
    masked = img_data[mask_data > 0]
    landmarks = get_landmarks(masked, landmark_percs)
    f = interp1d(landmarks, standard_scale, fill_value='extrapolate')
    normed = f(img_data)
    return nib.Nifti1Image(normed, img.affine, img.header)
