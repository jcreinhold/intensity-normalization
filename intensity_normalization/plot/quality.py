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


def plot_pairwise_jsd(img_dir, mask_dir, outfn='pairwisejsd.png'):
    """
    create a figure of pairwise jensen-shannon divergence for all images in a directory

    Args:
        img_dir (str): path to directory of nifti images
        mask_dir (str): path to directory of corresponding masks

    Returns:
        None (saves a figure)
    """
    pairwise_jsd = quality.pairwise_jsd(img_dir, mask_dir) * 10e4
    plt.hist(pairwise_jsd)
    plt.xlabel(r'Jensen-Shannon Divergence ($\times 10^{4}$)')
    plt.ylabel('Count')
    plt.title(
        r'Histogram of Pairwise JSD — $\mu$ = ' + f'{np.mean(pairwise_jsd):.2e}' + r' $\sigma$ = ' + f'{np.std(pairwise_jsd):.2e}')
    plt.savefig(outfn, transparent=True, dpi=200)
