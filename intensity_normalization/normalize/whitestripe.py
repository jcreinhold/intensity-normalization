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
    Use histogram matching method ([1,2]) to normalize the intensities of a set of MR images

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
        normalized (np.ndarray): set of normalized images from img_dir

    References:
        [1] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee,
            F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
            D. S. Reich, and C. M. Crainiceanu, “Statistical normalization
            techniques for magnetic resonance imaging,” NeuroImage Clin.,
            vol. 6, pp. 9–19, 2014.
    """
    data = glob(os.path.join(img_dir, '*.nii*'))
    if mask_dir is None:
        masks = [NULL] * len(data)
    else:
        masks = glob(os.path.join(mask_dir, '*.nii*'))
        if len(data) != len(masks):
            NormalizationError('Number of images and masks must be equal, Images: {}, Masks: {}'
                               .format(len(data), len(masks)))
    if output_dir is None:
        output_files = [NULL] * len(data)
    else:
        out_fns = []
        for fn in data:
            _, base, ext = io.split_filename(fn)
            out_fns.append(os.path.join(output_dir, base, ext))
        output_files = StrVector(out_fns)
    for img_fn, mask_fn, output_fn in zip(data, masks, output_files):
        img = nb.check_nifti(img_fn, reorient=False, allow_array=False)
        if mask_fn is NULL:
            brain = img
        else:
            mask = nb.check_nifti(mask_fn, reorient=False, allow_array=False)
            brain = nb.mask_img(img, mask)
        indices = ws.whitestripe(brain, type=contrast, slices=IntVector(range(slices[0],slices[1])),
                                 verbose=False)
        brain = ws.whitestripe_norm(brain, indices[0])
        if write_to_disk:
            nb.write_nifti(brain, output_fn)
    normalized = np.array(brain)
    return normalized
