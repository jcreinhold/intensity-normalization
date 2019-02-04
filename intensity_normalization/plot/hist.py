#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.plot.hist

plot histogram of one img or all imgs in directory

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 21, 2018
"""

import logging
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from intensity_normalization.errors import NormalizationError
from intensity_normalization.utilities.io import glob_nii

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    sns.set(style='whitegrid', font_scale=2, rc={'grid.color': '.9'})
except ImportError:
    logger.debug("Seaborn not installed, so plots won't look as pretty :-(")


def all_hists(img_dir, mask_dir=None, alpha=0.8, figsize=(12,10), **kwargs):
    """
    plot all histograms over one another to get an idea of the
    spread for a sample/population

    note that all hsitograms are for the intensities within a given brain mask
    or estimated foreground mask (the estimate is just all intensities above the mean)

    Args:
        img_dir (str): path to images
        mask_dir (str): path to corresponding masks of imgs
        alpha (float): controls alpha parameter of individual line plots (default: 0.8)
        figsize (tuple): size of figure (default: (12,10))
        **kwargs: for numpy histogram routine

    Returns:
        ax (matplotlib.axes.Axes): plotted on ax obj
    """
    imgs = glob_nii(img_dir)
    if mask_dir is not None:
        masks = glob_nii(mask_dir)
    else:
        masks = [None] * len(imgs)
    if len(imgs) != len(masks):
        raise NormalizationError('Number of images and masks must be equal ({:d} != {:d})'
                                 .format(len(imgs), len(masks)))
    _, ax = plt.subplots(figsize=figsize)
    for i, (img_fn, mask_fn) in enumerate(zip(imgs, masks), 1):
        logger.info('Creating histogram for image {:d}/{:d}'.format(i,len(imgs)))
        img = nib.load(img_fn)
        if mask_fn is not None:
            mask = nib.load(mask_fn)
        else:
            mask = None
        _ = hist(img, mask, ax=ax, alpha=alpha, **kwargs)
    ax.set_xlabel('Intensity')
    ax.set_ylabel(r'Log$_{10}$ Count')
    ax.set_ylim((0, None))
    return ax


def hist(img, mask=None, ax=None, n_bins=200, log=True, alpha=0.8, lw=3, **kwargs):
    """
    plots the histogram of an ants object (line histogram) within a given brain mask
    or estimated foreground mask (the estimate is just all intensities above the mean)

    Args:
        img (nibabel.nifti1.Nifti1Image): MR image of interest
        mask (nibabel.nifti1.Nifti1Image): brain mask of img (default: None)
        ax (matplotlib.axes.Axes): ax to plot on (default: None)
        n_bins (int): number of bins to use in histogram (default: 200)
        log (bool): use log scale (default: True)
        alpha (float): value in [0,1], controls opacity of line plot
        kwargs (dict): arguments to numpy histogram func

    Returns:
        ax (matplotlib.axes.Axes): plotted on ax obj
    """
    if ax is None:
        _, ax = plt.subplots()
    data = img.get_data()[mask.get_data()==1] if mask is not None else img.get_data()
    hist_, bin_edges = np.histogram(data.flatten(), n_bins, **kwargs)
    bins = np.diff(bin_edges)/2 + bin_edges[:-1]
    if log:
        # catch divide by zero warnings in call to log
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            hist_ =  np.log10(hist_)
            hist_[hist_ == -np.inf] = 0
    ax.plot(bins, hist_, alpha=alpha, linewidth=lw)
    return ax

