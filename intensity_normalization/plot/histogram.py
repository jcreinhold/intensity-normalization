"""Plot histogram of the intensities of a set of images
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 02 Jun 2021
"""

from __future__ import annotations

__all__ = ["HistogramPlotter", "plot_histogram"]

import argparse
import collections.abc
import logging
import pathlib
import typing
import warnings

import matplotlib.pyplot as plt
import numpy as np

import intensity_normalization as intnorm
import intensity_normalization.base_cli as intnormcli
import intensity_normalization.typing as intnormt
import intensity_normalization.util.io as intnormio

logger = logging.getLogger(__name__)

try:
    import seaborn as sns

    sns.set(style="whitegrid", font_scale=2, rc={"grid.color": ".9"})
except ImportError:
    logger.debug("Seaborn not installed. Plots won't look as pretty.")
    sns = None


class HistogramPlotter(intnormcli.DirectoryCLI):
    def __init__(
        self,
        *,
        figsize: tuple[int, int] = (12, 10),
        alpha: float = 0.8,
        title: str | None = None,
    ):
        super().__init__()
        self.figsize = figsize
        self.alpha = alpha
        self.title = title

    def __call__(
        self,
        images: collections.abc.Sequence[intnormt.ImageLike],
        /,
        masks: collections.abc.Sequence[intnormt.ImageLike] | None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        **kwargs: typing.Any,
    ) -> plt.Axes:
        if not images:
            raise ValueError("'images' must be a non-empty sequence.")
        if hasattr(images[0], "get_fdata"):
            images = [img.get_fdata() for img in images]  # type: ignore[attr-defined]
        if masks is not None:
            if hasattr(masks[0], "get_fdata"):
                masks = [msk.get_fdata() for msk in masks]  # type: ignore[attr-defined]
            if len(images) != len(masks):
                raise ValueError("Number of images and masks must be equal.")
        ax = self.plot_all_histograms(images, masks, **kwargs)
        return ax

    def plot_all_histograms(
        self,
        images: collections.abc.Sequence[intnormt.ImageLike],
        /,
        masks: collections.abc.Sequence[intnormt.ImageLike] | None,
        **kwargs: typing.Any,
    ) -> plt.Axes:
        _, ax = plt.subplots(figsize=self.figsize)
        n_images = len(images)
        for i, (image, mask) in enumerate(intnormio.zip_with_nones(images, masks), 1):
            logger.info(f"Plotting histogram ({i:d}/{n_images:d}).")
            _ = plot_histogram(image, mask, ax=ax, alpha=self.alpha, **kwargs)
        ax.set_xlabel("Intensity")
        ax.set_ylabel(r"Log$_{10}$ Count")
        ax.set_ylim((0, None))
        if self.title is not None:
            ax.set_title(self.title)
        return ax

    def from_directories(
        self,
        image_dir: intnormt.PathLike,
        /,
        mask_dir: intnormt.PathLike | None = None,
        *,
        ext: str = "nii*",
        exclude: collections.abc.Sequence[str] = ("membership",),
        **kwargs: typing.Any,
    ) -> plt.Axes:
        images, masks = intnormio.gather_images_and_masks(
            image_dir, mask_dir, ext=ext, exclude=exclude
        )
        return self(images, masks, **kwargs)

    @staticmethod
    def name() -> str:
        return "hist"

    @staticmethod
    def fullname() -> str:
        return "Histogram plotter"

    @staticmethod
    def description() -> str:
        return "Plot the histogram of an image."

    @classmethod
    def get_parent_parser(
        cls,
        desc: str,
        valid_modalities: frozenset[str] = intnorm.VALID_MODALITIES,
        **kwargs: typing.Any,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "image_dir",
            type=intnormt.dir_path(),
            help="Path of directory containing images for which to plot histograms.",
        )
        parser.add_argument(
            "-m",
            "--mask-dir",
            type=intnormt.dir_path(),
            default=None,
            help="Path of directory to corresponding foreground masks for image_dir.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=intnormt.save_file_path(),
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
            type=intnormt.probability_float(),
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
            "-e",
            "--extension",
            type=str,
            default="nii*",
            help="Extension of images.",
        )
        parser.add_argument(
            "-exc",
            "--exclude",
            nargs="+",
            default=["membership"],
            help="Exclude filenames including these strings.",
        )
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            default=0,
            help="Increase output verbosity (e.g., -vv is more than -v).",
        )
        parser.add_argument(
            "--version",
            action="store_true",
            help="Print the version of intensity-normalization.",
        )
        return parser

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace) -> HistogramPlotter:
        return cls(figsize=args.figsize, alpha=args.alpha, title=args.title)

    def call_from_argparse_args(
        self, args: argparse.Namespace, /, **kwargs: typing.Any
    ) -> None:
        _ = self.from_directories(
            args.image_dir, args.mask_dir, ext=args.extension, exclude=args.exclude
        )
        if args.output is None:
            args.output = pathlib.Path(args.image_dir).resolve() / "hist.pdf"
        logger.info(f"Saving histogram: {args.output}.")
        plt.savefig(args.output)


def plot_histogram(
    image: intnormt.ImageLike,
    /,
    mask: intnormt.ImageLike | None = None,
    *,
    ax: plt.Axes | None = None,
    n_bins: int = 200,
    log: bool = True,
    alpha: float = 0.8,
    linewidth: float = 3.0,
    **kwargs: typing.Any,
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
        linewidth: width of line in histogram plot
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
