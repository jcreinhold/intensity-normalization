"""Foreground histogram plotting for validating normalization results.

matplotlib is an optional dependency (``pip install intensity-normalization[plot]``),
imported lazily inside the function.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence
from os import PathLike

from intensity_normalization import _image, histogram
from intensity_normalization._image import ImageLike
from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["plot_histograms"]


def plot_histograms(
    images: Sequence[ImageLike],
    /,
    masks: Sequence[ImageLike | None] | None = None,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    log_scale: bool = True,
    output: str | PathLike[str] | None = None,
    seed: int | None = 0,
) -> typing.Any:
    """Plot smoothed foreground-intensity histograms of a set of images.

    The recommended way to validate normalization: run before and after and
    compare. Histograms are kernel density estimates (see
    :mod:`intensity_normalization.histogram`).

    Args:
        images: images to plot (numpy arrays or nibabel images).
        masks: optional foreground (brain) mask per image.
        labels: legend labels; defaults to ``image 0``, ``image 1``, ...
        title: plot title.
        log_scale: plot the density on a log scale (default; makes tissue
            peaks of MR brain images easier to compare).
        output: save the figure to this path instead of showing it.
        seed: RNG seed for the KDE subsample; ``None`` is nondeterministic.

    Returns:
        The matplotlib ``Figure``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exn:
        msg = "Plotting requires matplotlib. Install it with: pip install 'intensity-normalization[plot]'"
        raise IntensityNormalizationError(msg) from exn

    if len(images) == 0:
        raise IntensityNormalizationError("No images provided to plot.")
    if masks is not None and len(masks) != len(images):
        raise ValueError(f"Got {len(images)} images but {len(masks)} masks.")

    fig, ax = plt.subplots()
    for i, image in enumerate(images):
        data, _ = _image.unwrap(image)
        mask = masks[i] if masks is not None else None
        mask_data = _image.unwrap_mask(image, mask)
        foreground = _image.foreground_values(data, mask_data)
        grid, pdf = histogram.smooth_histogram(foreground, seed=seed)
        label = labels[i] if labels is not None else f"image {i}"
        ax.plot(grid, pdf, label=label)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("intensity")
    ax.set_ylabel("density")
    ax.legend()
    if title:
        ax.set_title(title)
    if output is not None:
        fig.savefig(output, bbox_inches="tight")
    return fig
