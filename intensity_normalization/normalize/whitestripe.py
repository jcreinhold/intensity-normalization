#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.whitestripe

Use the White Stripe method outlined in [1] to normalize
the intensity of an MR image

Note that this package requires RAVEL (and its dependencies)
to be installed in R

References:
﻿   [1] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee,
        F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
        D. S. Reich, and C. M. Crainiceanu, “Statistical normalization
        techniques for magnetic resonance imaging,” NeuroImage Clin.,
        vol. 6, pp. 9–19, 2014.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Apr 27, 2018
"""

from __future__ import print_function, division

from glob import glob
import logging
import os

import numpy as np
from rpy2.robjects.vectors import StrVector, IntVector
from rpy2.robjects.packages import importr
from rpy2.rinterface import NULL

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities import io

nb = importr('neurobase')
ws = importr('WhiteStripe')

logger = logging.getLogger(__name__)


def ws_normalize(img_dir, contrast, mask_dir=None, output_dir=None, write_to_disk=True, slices=(80, 120)):
    """
    Use WhiteStripe normalization method ([1]) to normalize the intensities of
    a set of MR images by normalizing an area around the white matter peak of the histogram

    Args:
        img_dir (str): directory containing MR images to be normalized
        contrast (str): contrast of MR images to be normalized (T1, T2, FLAIR or PD)
        mask_dir (str): if images are not skull-stripped, then provide brain mask
        output_dir (str): directory to save images if you do not want them saved in
            same directory as img_dir
        write_to_disk (bool): write the normalized data to disk or nah
        slices (tuple): two ints in tuple corresponding to region from which to sample
            for the whitestripe procedure

    Returns:
        normalized (np.ndarray): last normalized image data from img_dir
            I know this is an odd behavior, but yolo

    References:
        [1] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee,
            F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
            D. S. Reich, and C. M. Crainiceanu, “Statistical normalization
            techniques for magnetic resonance imaging,” NeuroImage Clin.,
            vol. 6, pp. 9–19, 2014.
    """

    # grab the file names for the images of interest
    data = sorted(glob(os.path.join(img_dir, '*.nii*')))

    # define and get the brain masks for the images, if defined
    if mask_dir is None:
        masks = [NULL] * len(data)
    else:
        masks = sorted(glob(os.path.join(mask_dir, '*.nii*')))
        if len(data) != len(masks):
            NormalizationError('Number of images and masks must be equal, Images: {}, Masks: {}'
                               .format(len(data), len(masks)))

    # define the output directory and corresponding output file names
    if output_dir is None:
        output_files = [NULL] * len(data)
    else:
        out_fns = []
        for fn in data:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + ext))
        output_files = StrVector(out_fns)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # define the slices to use for whitestripe analysis if defined, otherwise use whole img
    if slices is not None and len(slices) == 2:
        ws_slices = IntVector(range(slices[0], slices[1]))
    else:
        ws_slices = NULL

    # control verbosity of output when making whitestripe function call
    if 0 < logger.getEffectiveLevel() <= logging.getLevelName('DEBUG'):
        verbose = True
    else:
        verbose = False

    # do whitestripe normalization and save the results
    for i, (img_fn, mask_fn, output_fn) in enumerate(zip(data, masks, output_files), 1):
        logger.info('Normalizing image: {} ({:d}/{:d})'.format(img_fn, i, len(data)))
        img = nb.check_nifti(img_fn, reorient=False, allow_array=False)
        if mask_fn is NULL:
            brain = img
        else:
            mask = nb.check_nifti(mask_fn, reorient=False, allow_array=False)
            brain = nb.mask_img(img, mask)
        indices = ws.whitestripe(brain, type=contrast, slices=ws_slices, verbose=verbose)
        img = ws.whitestripe_norm(img, indices[0])
        if write_to_disk:
            logger.info('Saving normalized image: {} ({:d}/{:d})'.format(output_fn, i, len(data)))
            nb.write_nifti(img, output_fn)

    # output the last normalized image (mostly for testing purposes)
    normalized = np.array(img.slots['.Data'])
    return normalized
