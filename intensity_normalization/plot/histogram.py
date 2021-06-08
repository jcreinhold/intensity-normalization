# -*- coding: utf-8 -*-
"""
intensity_normalization.plot.histogram

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 02, 2021
"""

from argparse import ArgumentParser, Namespace
import logging
from typing import List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np

from intensity_normalization.parse import CLI
from intensity_normalization.type import Array, PathLike
from intensity_normalization.util.io import gather_images_and_masks

logger = logging.getLogger(__name__)

try:
    import seaborn as sns

    sns.set(style="whitegrid", font_scale=2, rc={"grid.color": ".9"})
except ImportError:
    logger.debug("Seaborn not installed. Plots won't look as pretty.")


class HistogramPlotter(CLI):
    def __init__(
        self,
        images: List[Array],
        masks: List[Optional[Array]],
        figsize: Tuple[int, int] = (12, 10),
        alpha: float = 0.8,
    ):
        self.images = images
        self.masks = masks
        self.figsize = figsize
        self.alpha = alpha

    @classmethod
    def from_directories(
        cls,
        image_dir: PathLike,
        mask_dir: Optional[PathLike] = None,
        ext: str = "nii*",
        **kwargs,
    ):
        images, masks = gather_images_and_masks(image_dir, mask_dir, ext, True)
        return cls(images, masks, **kwargs)

    def plot_all_histograms(self, **kwargs):
        _, ax = plt.subplots(figsize=self.figsize)
        n_images = len(self.images)
        for i, (image, mask) in enumerate(zip(self.images, self.masks), 1):
            logger.info(f"Creating histogram ({i:d}/{n_images:d})")
            _ = plot_histogram(image, mask, ax=ax, alpha=self.alpha, **kwargs)
        ax.set_xlabel("Intensity")
        ax.set_ylabel(r"Log$_{10}$ Count")
        ax.set_ylim((0, None))
        return ax

    @staticmethod
    def description() -> str:
        return "Plot the histogram of an image."

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        raise NotImplementedError

    @classmethod
    def from_argparse_args(cls, args: Namespace):
        raise NotImplementedError


def plot_histogram(
    image: Array,
    mask: Optional[Array] = None,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 200,
    log: bool = True,
    alpha: float = 0.8,
    linewidth: float = 3.0,
    **kwargs,
):
    """
    plots the histogram of the intensities of a numpy array within a given brain mask
    or estimated foreground mask (the estimate is just all intensities above the mean)

    Args:
        image: image/array of interest
        mask: mask of the foreground, if none then assume skull-stripped
        ax: matplotlib ax to plot on, if none then create new ax
        n_bins: number of bins to use in histogram
        log: use log scale on the y-axis
        alpha: value in [0,1], controls opacity of line plot
        kwargs: arguments to the histogram function

    Returns:
        ax: the ax the histogram is plotted on
    """
    if ax is None:
        _, ax = plt.subplots()
    data = image[mask > 0.0] if mask is None else image[image > 0.0]
    hist, bin_edges = np.histogram(data.flatten(), n_bins, **kwargs)
    bins = np.diff(bin_edges) / 2 + bin_edges[:-1]
    if log:
        # catch divide by zero warnings in call to log
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            hist = np.log10(hist)
            hist[np.isinf(hist)] = 0.0
    ax.plot(bins, hist, alpha=alpha, linewidth=linewidth)
    return ax
