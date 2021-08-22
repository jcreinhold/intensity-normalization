# -*- coding: utf-8 -*-
"""
intensity_normalization.plot.histogram

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 02, 2021
"""

import logging
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from intensity_normalization.parse import CLIParser
from intensity_normalization.type import (
    Array,
    ArrayOrNifti,
    PathLike,
    dir_path,
    probability_float,
    save_file_path,
)
from intensity_normalization.util.io import gather_images_and_masks

logger = logging.getLogger(__name__)

try:
    import seaborn as sns

    sns.set(style="whitegrid", font_scale=2, rc={"grid.color": ".9"})
except ImportError:
    logger.debug("Seaborn not installed. Plots won't look as pretty.")

HP = TypeVar("HP", bound="HistogramPlotter")


class HistogramPlotter(CLIParser):
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        alpha: float = 0.8,
        title: Optional[str] = None,
    ):
        self.figsize = figsize
        self.alpha = alpha
        self.title = title

    def __call__(  # type: ignore[no-untyped-def,override]
        self,
        images: List[ArrayOrNifti],
        masks: List[Optional[ArrayOrNifti]],
        **kwargs,
    ) -> plt.Axes:
        assert len(images) > 0
        assert len(images) == len(masks)
        if hasattr(images[0], "get_fdata"):
            images = [img.get_fdata() for img in images]
        if hasattr(masks[0], "get_fdata"):
            masks = [msk.get_fdata() for msk in masks]  # type: ignore[union-attr]
        ax = self.plot_all_histograms(images, masks, **kwargs)
        return ax

    def plot_all_histograms(  # type: ignore[no-untyped-def]
        self,
        images: List[Array],
        masks: List[Optional[Array]],
        **kwargs,
    ) -> plt.Axes:
        _, ax = plt.subplots(figsize=self.figsize)
        n_images = len(images)
        for i, (image, mask) in enumerate(zip(images, masks), 1):
            logger.info(f"Creating histogram ({i:d}/{n_images:d})")
            _ = plot_histogram(image, mask, ax=ax, alpha=self.alpha, **kwargs)
        ax.set_xlabel("Intensity")
        ax.set_ylabel(r"Log$_{10}$ Count")
        ax.set_ylim((0, None))
        if self.title is not None:
            ax.set_title(self.title)
        return ax

    def from_directories(  # type: ignore[no-untyped-def]
        self,
        image_dir: PathLike,
        mask_dir: Optional[PathLike] = None,
        ext: str = "nii*",
        **kwargs,
    ) -> plt.Axes:
        images, masks = gather_images_and_masks(image_dir, mask_dir, ext, True)
        return self(images, masks, **kwargs)

    @staticmethod
    def description() -> str:
        return "Plot the histogram of an image."

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        parser = ArgumentParser(
            description=desc,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "image_dir",
            type=dir_path(),
            help="Path of image directory to plot histograms for.",
        )
        parser.add_argument(
            "-m",
            "--mask-dir",
            type=dir_path(),
            default=None,
            help="Path of directory to corresponding foreground masks for image_dir.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=save_file_path(),
            default=None,
            help="Path to save histogram.",
        )
        parser.add_argument(
            "-fs",
            "--figsize",
            nargs=2,
            type=int,
            default=(12, 10),
            help="Figure size of histogram.",
        )
        parser.add_argument(
            "-a",
            "--alpha",
            type=probability_float,  # type: ignore[arg-type]
            default=0.8,
            help="Alpha level for line representing histogram.",
        )
        parser.add_argument(
            "-t",
            "--title",
            type=str,
            default=None,
            help="Title for histogram plot.",
        )
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            default=0,
            help="Increase output verbosity (e.g., -vv is more than -v).",
        )
        return parser

    @classmethod
    def from_argparse_args(cls: Type[HP], args: Namespace) -> HP:
        return cls(args.figsize, args.alpha, args.title)

    def call_from_argparse_args(self, args: Namespace) -> None:
        _ = self.from_directories(args.image_dir, args.mask_dir)
        if args.output is None:
            args.output = Path.cwd().resolve() / "hist.pdf"
        logger.info(f"Saving histogram: {args.output}")
        plt.savefig(args.output)


def plot_histogram(  # type: ignore[no-untyped-def]
    image: Array,
    mask: Optional[Array] = None,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 200,
    log: bool = True,
    alpha: float = 0.8,
    linewidth: float = 3.0,
    **kwargs,
) -> plt.Axes:
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
    data = image[image > image.mean()] if mask is None else image[mask > 0.0]
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
