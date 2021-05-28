#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.whitestripe

Use the White Stripe method outlined in [1] to normalize
the intensity of an MR image

References:
﻿   [1] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee,
        F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
        D. S. Reich, and C. M. Crainiceanu, “Statistical normalization
        techniques for magnetic resonance imaging,” NeuroImage Clin.,
        vol. 6, pp. 9–19, 2014.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 27, 2018
"""

import logging
import os

import nibabel as nib
import numpy as np

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import io, hist

logger = logging.getLogger(__name__)


def ws_normalize(
    image_dir, modality, mask_dir=None, output_dir=None, write_to_disk=True
):
    """
    Use WhiteStripe normalization method ([1]) to normalize the intensities of
    a set of MR images by normalizing an area around the white matter peak of the histogram

    Args:
        image_dir (str): directory containing MR images to be normalized
        modality (str): contrast of MR images to be normalized  (e.g., t1)
        mask_dir (str): if images are not skull-stripped, then provide brain mask
        output_dir (str): directory to save images if you do not want them saved in
            same directory as image_dir
        write_to_disk (bool): write the normalized data to disk or nah

    Returns:
        normalized (np.ndarray): last normalized image data from image_dir
            (This is really for debugging purposes.)

    References:
        [1] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee,
            F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
            D. S. Reich, and C. M. Crainiceanu, “Statistical normalization
            techniques for magnetic resonance imaging,” NeuroImage Clin.,
            vol. 6, pp. 9–19, 2014.
    """

    # grab the file names for the images of interest
    data = io.glob_nii(image_dir)

    # define and get the brain masks for the images, if defined
    if mask_dir is None:
        masks = [None] * len(data)
    else:
        masks = io.glob_nii(mask_dir)
        if len(data) != len(masks):
            raise NormalizationError(
                "Number of images and masks must be equal, Images: {}, Masks: {}".format(
                    len(data), len(masks)
                )
            )

    # define the output directory and corresponding output file names
    if output_dir is None:
        output_files = [None] * len(data)
    else:
        output_files = []
        for fn in data:
            _, base, ext = io.split_filename(fn)
            output_files.append(os.path.join(output_dir, base + "_ws" + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    normalized = None
    # do whitestripe normalization and save the results
    for i, (image_fn, mask_fn, output_fn) in enumerate(
        zip(data, masks, output_files), 1
    ):
        logger.info("Normalizing image: {} ({:d}/{:d})".format(image_fn, i, len(data)))
        image = io.open_nii(image_fn)
        mask = io.open_nii(mask_fn) if mask_fn is not None else None
        indices = whitestripe(image, modality, mask=mask)
        normalized = whitestripe_norm(image, indices)
        if write_to_disk:
            logger.info(
                "Saving normalized image: {} ({:d}/{:d})".format(
                    output_fn, i, len(data)
                )
            )
            io.save_nii(normalized, output_fn)

    # output the last normalized image (mostly for testing purposes)
    return normalized


def whitestripe(image, modality, mask=None, width=0.05, width_l=None, width_u=None):
    """
    find the "(normal appearing) white (matter) stripe" of the input MR image
    and return the indices

    Args:
        image (nibabel.nifti1.Nifti1Image): target MR image
        modality (str): contrast of img (e.g., T1)
        mask (nibabel.nifti1.Nifti1Image): brainmask for img (None is default, for skull-stripped img)
        width (float): width quantile for the "white (matter) stripe"
        width_l (float): lower bound for width (default None, derives from width)
        width_u (float): upper bound for width (default None, derives from width)

    Returns:
        ws_ind (np.ndarray): the white stripe indices (boolean mask)
    """
    if width_l is None and width_u is None:
        width_l = width
        width_u = width
    data = image.get_fdata()
    if mask is not None:
        mask_data = mask.get_fdata()
        masked = data * mask_data  # noqa
        voi = data[mask_data > 0.0]  # noqa
    else:
        masked = data
        foreground_estimate = data > data.mean()  # noqa
        voi = data[foreground_estimate]  # noqa
    mode = hist.get_tissue_mode(voi, modality)
    img_mode_q = np.mean(voi < mode)
    ws = np.percentile(
        voi, (max(img_mode_q - width_l, 0) * 100, min(img_mode_q + width_u, 1) * 100)
    )
    ws_ind = np.logical_and(masked > ws[0], masked < ws[1])
    if len(ws_ind) == 0:
        raise NormalizationError("WhiteStripe failed to find any valid indices!")
    return ws_ind


def whitestripe_norm(img, indices):
    """
    use the whitestripe indices to standardize the data (i.e., subtract the
    mean of the values in the indices and divide by the std of those values)

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        indices (np.ndarray): whitestripe indices (see whitestripe func)

    Returns:
        norm_img (nibabel.nifti1.Nifti1Image): normalized image in nifti format
    """
    img_data = img.get_fdata()
    mu = np.mean(img_data[indices])
    sig = np.std(img_data[indices])
    norm_img_data = (img_data - mu) / sig
    norm_img = nib.Nifti1Image(norm_img_data, img.affine, img.header)
    return norm_img
