#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.plot.quality

create plots measuring quality

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 04, 2018
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from intensity_normalization.utilities import quality

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    sns.set(style='white', font_scale=2, rc={'figure.figsize': (12, 10)})
except ImportError:
    logger.debug("Seaborn not installed, so plots won't look as pretty :-(")


def plot_pairwise_jsd(img_dir, mask_dir, outfn='pairwisejsd.png', nbins=200, fit_exp=True):
    """
    create a figure of pairwise jensen-shannon divergence for all images in a directory

    Args:
        img_dir (str): path to directory of nifti images
        mask_dir (str): path to directory of corresponding masks

    Returns:
        None (saves a figure)
    """
    pairwise_jsd = quality.pairwise_jsd(img_dir, mask_dir, nbins=nbins)
    _, ax = plt.subplots(1, 1)
    ax.hist(pairwise_jsd, label='Hist.', density=True)
    if fit_exp:
        from scipy.stats import expon
        loc, scale = expon.fit(pairwise_jsd, floc=0)
        x = np.linspace(0, np.max(pairwise_jsd))
        ax.plot(x, expon.pdf(x, loc, scale), lw=3, label='Exp. Fit')
        ax.legend()
        textstr = r'$\lambda = $' + f'{1/scale:.02e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.72, 0.80, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
    ax.set_xlabel(r'Jensen-Shannon Divergence')
    ax.set_ylabel('Density')
    ax.set_title(
        r'Density of Pairwise JSD — $\mu$ = ' + f'{np.mean(pairwise_jsd):.2e}' + r' $\sigma$ = ' + f'{np.std(pairwise_jsd):.2e}',
        pad=20)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.savefig(outfn, transparent=True, dpi=200)
