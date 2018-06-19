#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.hm

Use the method of Nyul and Udupa [1] (updated in [2])
to do histogram matching intensity normalization on a
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

from intensity_normalization.utilities import io

logger = logging.getLogger(__name__)


def hm_normalize(img_dir, mask_dir=None, output_dir=None, write_to_disk=True):
    """
    Use histogram matching method ([1,2]) to normalize the intensities of a set of MR images

    Args:
        img_dir (str): directory containing MR images
        img_dir (str): directory containing masks for MR images
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
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
            out_fns.append(os.path.join(output_dir, base + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    mask_files = [None] * len(input_files) if mask_dir is None else io.glob_nii(mask_dir)

    logger.info('Learning standard scale for the set of images')
    landmarks, pts = train(input_files, mask_files)

    for i, (img_fn, mask_fn, out_fn) in enumerate(zip(input_files, mask_files, out_fns)):
        _, base, _ = io.split_filename(img_fn)
        logger.info('Transforming image {} to standard scale ({:d}/{:d})'.format(base, i+1, len(input_files)))
        img = io.open_nii(img_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        normalized = do_hist_norm(img, pts, landmarks, mask)
        if write_to_disk:
            io.save_nii(normalized, out_fn, is_nii=True)

    return normalized


def train(img_fns, mask_fns=None, i_min=1, i_max=99, i_s_min=0, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
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
        m (np.ndarray): average landmark intensity for images
        h (np.ndarray): corresponding landmark points (i.e., the domain of m)
    """
    mask_fns = [None] * len(img_fns) if mask_fns is None else mask_fns
    h = np.arange(l_percentile, u_percentile+1, step)  # percentile landmarks
    ms = np.zeros(len(h))
    for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns)):
        img = io.open_nii(img_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        ms += get_landmarks(img, h, mask, i_min, i_max, i_s_min, i_s_max)
    m = ms / len(img_fns)
    return m, h


def get_landmarks(img, h, mask=None, i_min=1, i_max=99, i_s_min=0, i_s_max=100):
    """
    get the landmarks for Nyul and Udupa for a specific image

    Args:
        img (nibabel.nifti1.Nifti1Image): image on which to find landmarks
        h (np.ndarray): corresponding landmark points (i.e., the domain of m)
        mask (nibabel.nifti1.Nifti1Image): foreground mask for img
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale

    Returns:
        landmarks (np.ndarray): intensity values corresponding to h in img (w/ mask)
    """
    img_data = img.get_data()
    mask_data = img_data > img_data.mean() if mask is None else mask.get_data()
    masked = img_data[mask_data > 0]
    min_p = np.percentile(masked, i_min)
    max_p = np.percentile(masked, i_max)
    scaled_data = ((img_data - min_p + i_s_min) / max_p) * i_s_max
    landmarks = np.percentile(scaled_data[mask_data > 0], h)
    return landmarks


def do_hist_norm(img, h, m, mask=None, i_min=1, i_max=99, i_s_min=0, i_s_max=100):
    """
    do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks

    Args:
        img (nibabel.nifti1.Nifti1Image): image on which to find landmarks
        h (np.ndarray): corresponding landmark points (i.e., the domain of m)
        m (np.ndarray): landmarks on the standard scale
        mask (nibabel.nifti1.Nifti1Image): foreground mask for img
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): normalized image
    """
    img_data = img.get_data()
    mask_data = img_data > img_data.mean() if mask is None else mask.get_data()
    percentiles = np.concatenate((np.array([i_min]), h, np.array([i_max])))
    masked = img_data[mask_data > 0]
    m_obs = np.percentile(masked, percentiles)
    m_withends = np.concatenate((np.array([i_s_min]), m, np.array([i_s_max])))
    normed = img_data

    l_thresh = np.percentile(masked, i_min)
    u_thresh = np.percentile(masked, i_max)
    normed[normed <= l_thresh] = i_s_min
    normed[normed >= u_thresh] = i_s_max

    assert len(m_obs) == len(m_withends)

    for i in range(len(m_obs)-1):
        obs0 = m_obs[i]
        obs1 = m_obs[i + 1]
        m1 = m_withends[i + 1]
        m0 = m_withends[i]
        inds = np.logical_and(img_data < obs1, img_data >= obs0)
        normed[inds] = (((img_data[inds] - obs0) / (obs1 - obs0)) * (m1 - m0)) + m0

    return nib.Nifti1Image(normed, img.affine, img.header)
