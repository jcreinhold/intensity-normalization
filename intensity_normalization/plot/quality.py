#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.plot.quality

create plots measuring quality of normalization on a set of images

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

    sns.set(style="white", font_scale=2, rc={"figure.figsize": (12, 10)})
except ImportError:
    logger.debug("Seaborn not installed, so plots won't look as pretty :-(")


def plot_pairwise_jsd(
    image_dir, mask_dir, filename="pairwisejsd.png", n_bins=200, fit_chi2=True
):
    """
    create a figure of pairwise Jensen-Shannon Divergence (JSD) for all
    images in a directory

    Args:
        image_dir (str): path to directory of nifti images
        mask_dir (str): path to directory of corresponding masks
        filename (str): save plot to this filename
        n_bins (int): use this many bins in histogram
        fit_chi2 (bool): fit a chi2 distribution to the JSD

    Returns:
        ax (matplotlib ax): ax the plot was created on
    """
    pairwise_jsd = quality.pairwise_jsd(image_dir, mask_dir, nbins=n_bins)
    _, ax = plt.subplots(1, 1)
    ax.hist(pairwise_jsd, label="Hist.", density=True)
    if fit_chi2:
        from scipy.stats import chi2

        df, _, scale = chi2.fit(pairwise_jsd, floc=0)
        logger.info(f"df = {df:0.3e}, scale = {scale:0.3e}")
        x = np.linspace(0, np.max(pairwise_jsd), 200)
        ax.plot(x, chi2.pdf(x, df, scale=scale), lw=3, label=r"$\chi^2$ Fit")
        ax.legend()
        textstr = r"$df = $" + f"{df:0.2f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.72,
            0.80,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=props,
        )
    ax.set_xlabel(r"Jensen-Shannon Divergence")
    ax.set_ylabel("Density")
    ax.set_title(
        r"Density of Pairwise JSD — $\mu$ = "
        + f"{np.mean(pairwise_jsd):.2e}"
        + r" $\sigma$ = "
        + f"{np.std(pairwise_jsd):.2e}",
        pad=20,
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    if filename is not None:
        plt.savefig(filename, transparent=True, dpi=200)
    return ax
