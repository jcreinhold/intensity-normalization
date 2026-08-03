"""WhiteStripe normalization: standardize by the normal-appearing white matter statistics."""

from __future__ import annotations

import numpy as np

from intensity_normalization import _image, histogram
from intensity_normalization._image import BinaryMask, Image, IntensityArray, Mask
from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["whitestripe", "whitestripe_array"]


def whitestripe_array(
    data: IntensityArray,
    foreground: BinaryMask,
    *,
    peak: histogram.Peak,
    width: float = 0.05,
    width_l: float | None = None,
    width_u: float | None = None,
    norm_value: float = 1.0,
    seed: int | None = 0,
) -> IntensityArray:
    """WhiteStripe normalization of an intensity array (Shinohara et al., 2014).

    Finds the normal-appearing white matter (NAWM) as the intensities within
    ``width`` quantile around the white matter mode of the smoothed foreground
    histogram (the "white stripe"), then standardizes the array to the mean
    and standard deviation of that stripe, scaled by ``norm_value``.

    Args:
        data: intensity array.
        foreground: boolean foreground (brain) mask.
        peak: which histogram peak anchors the stripe ("last", "largest",
            "first"); resolve modality names with
            :func:`intensity_normalization.histogram.resolve_peak`.
        width: quantile half-width of the stripe around the tissue mode.
        width_l: asymmetric override for the lower width.
        width_u: asymmetric override for the upper width.
        norm_value: multiply the standardized array by this value.
        seed: RNG seed for the KDE subsample; ``None`` is nondeterministic.

    Returns:
        The normalized intensity array.
    """
    if width_l is None:
        width_l = width
    if width_u is None:
        width_u = width
    foreground_values = data[foreground]

    mode = histogram.tissue_mode(foreground_values, peak=peak, seed=seed)
    mode_quantile = float(np.mean(foreground_values < mode))
    lower = max(mode_quantile - width_l, 0.0)
    upper = min(mode_quantile + width_u, 1.0)
    ws_l, ws_u = np.quantile(foreground_values, (lower, upper))

    stripe = foreground & (data > ws_l) & (data < ws_u)
    stripe_values = data[stripe].astype(np.float64)
    if stripe_values.size == 0:
        msg = (
            "The white stripe is empty (no voxels within the intensity band "
            f"({ws_l:.3f}, {ws_u:.3f}) around the tissue mode {mode:.3f}). "
            "Check the modality/peak choice and the mask."
        )
        raise IntensityNormalizationError(msg)
    std = float(stripe_values.std())
    if std == 0.0:
        msg = "The white stripe has zero standard deviation; cannot normalize by it."
        raise IntensityNormalizationError(msg)
    return (data - stripe_values.mean()) / std * norm_value


def whitestripe(
    image: Image,
    mask: Mask | None = None,
    *,
    modality: str = "t1",
    peak: histogram.Peak | None = None,
    width: float = 0.05,
    width_l: float | None = None,
    width_u: float | None = None,
    norm_value: float = 1.0,
    seed: int | None = 0,
) -> Image:
    """WhiteStripe normalize an MR image (numpy or nibabel); see :func:`whitestripe_array`.

    Args:
        image: numpy array or nibabel image; the same type is returned.
        mask: foreground (brain) mask. If None, estimated as positive voxels.
        modality: one of "t1", "t2", "flair", "pd", "md", "other"; selects
            which histogram peak anchors the stripe.
        peak: explicit peak override ("last", "largest", "first") for
            non-standard data.
        width: quantile half-width of the stripe around the tissue mode.
        width_l: asymmetric override for the lower width.
        width_u: asymmetric override for the upper width.
        norm_value: multiply the standardized image by this value.
        seed: RNG seed for the KDE subsample; ``None`` is nondeterministic.

    Returns:
        The normalized image, same type as ``image``.
    """
    data, restore = _image.unwrap(image)
    foreground = _image.resolve_foreground(data, _image.unwrap_mask(image, mask))
    normalized = whitestripe_array(
        data,
        foreground,
        peak=histogram.resolve_peak(modality, peak),
        width=width,
        width_l=width_l,
        width_u=width_u,
        norm_value=norm_value,
        seed=seed,
    )
    return restore(normalized)
