"""Z-score normalization: standardize foreground intensities to zero mean, unit variance."""

from __future__ import annotations

import numpy as np

from intensity_normalization import _image
from intensity_normalization._image import BinaryMask, Image, IntensityArray, Mask
from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["zscore", "zscore_array"]


def zscore_array(
    data: IntensityArray,
    foreground: BinaryMask,
    *,
    norm_value: float = 1.0,
) -> IntensityArray:
    """Z-score normalize an intensity array by its foreground intensities.

    Subtracts the foreground mean and divides by the foreground standard
    deviation, then scales to ``norm_value``. Works for any anatomy/modality
    (not brain-specific).

    Args:
        data: intensity array.
        foreground: boolean foreground (brain) mask (see
            :func:`intensity_normalization._image.resolve_foreground`).
        norm_value: multiply the standardized array by this value.

    Returns:
        The normalized intensity array.
    """
    foreground64 = _image.foreground_values(data, foreground).astype(np.float64)
    # statistics in float64: float32 accumulation can make the std of
    # near-constant foregrounds spuriously nonzero
    std = float(foreground64.std())
    if std == 0.0:
        msg = "Foreground intensities have zero standard deviation; cannot z-score normalize."
        raise IntensityNormalizationError(msg)
    normalized = (data - foreground64.mean()) / std * norm_value
    return normalized.astype(np.float32)


def zscore(
    image: Image,
    mask: Mask | None = None,
    *,
    norm_value: float = 1.0,
) -> Image:
    """Z-score normalize an MR image (numpy or nibabel); see :func:`zscore_array`.

    Args:
        image: numpy array or nibabel image; the same type is returned.
        mask: foreground (brain) mask. If None, the foreground is estimated as
            positive voxels (i.e., the image is assumed skull-stripped).
        norm_value: multiply the standardized image by this value.

    Returns:
        The normalized image, same type as ``image``.
    """
    data, restore = _image.unwrap(image)
    foreground = _image.resolve_foreground(data, _image.unwrap_mask(image, mask))
    return restore(zscore_array(data, foreground, norm_value=norm_value))
