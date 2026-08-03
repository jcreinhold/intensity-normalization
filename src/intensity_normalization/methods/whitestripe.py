"""WhiteStripe normalization: standardize by the normal-appearing white matter statistics."""

from __future__ import annotations

import numpy as np

from intensity_normalization import _image, histogram
from intensity_normalization._image import ImageLike
from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["whitestripe"]


def whitestripe(
    image: ImageLike,
    /,
    mask: ImageLike | None = None,
    *,
    modality: str = "t1",
    peak: histogram.Peak | None = None,
    width: float = 0.05,
    width_l: float | None = None,
    width_u: float | None = None,
    norm_value: float = 1.0,
    seed: int | None = 0,
) -> ImageLike:
    """WhiteStripe normalization (Shinohara et al., 2014).

    Finds the normal-appearing white matter (NAWM) as the intensities within
    ``width`` quantile around the white matter mode of the smoothed foreground
    histogram (the "white stripe"), then standardizes the image to the mean
    and standard deviation of that stripe, scaled by ``norm_value``.

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
    if width_l is None:
        width_l = width
    if width_u is None:
        width_u = width
    data, restore = _image.unwrap(image)
    mask_data = _image.unwrap(mask)[0] if mask is not None else None
    foreground_mask = _image.get_mask(data, mask_data)
    foreground = data[foreground_mask]

    mode = histogram.tissue_mode(foreground, modality=modality, peak=peak, seed=seed)
    mode_quantile = float(np.mean(foreground < mode))
    lower = max(mode_quantile - width_l, 0.0)
    upper = min(mode_quantile + width_u, 1.0)
    ws_l, ws_u = np.quantile(foreground, (lower, upper))

    stripe = foreground_mask & (data > ws_l) & (data < ws_u)
    stripe_values = data[stripe]
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
    normalized = (data - stripe_values.mean()) / std * norm_value
    return restore(normalized)
