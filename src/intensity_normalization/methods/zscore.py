"""Z-score normalization: standardize foreground intensities to zero mean, unit variance."""

from __future__ import annotations

import numpy as np

from intensity_normalization import _image
from intensity_normalization._image import Image, IntensityArray, Mask
from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["zscore", "zscore_array"]


def zscore_array(
    data: IntensityArray,
    mask: IntensityArray | None = None,
    *,
    norm_value: float = 1.0,
) -> IntensityArray:
    """Z-score normalize an intensity array by its foreground intensities.

    Subtracts the foreground mean and divides by the foreground standard
    deviation, then scales to ``norm_value``. Works for any anatomy/modality
    (not brain-specific).

    Args:
        data: intensity array.
        mask: foreground (brain) mask array. If None, the foreground is
            estimated as positive voxels (i.e., the image is assumed
            skull-stripped).
        norm_value: multiply the standardized array by this value.

    Returns:
        The normalized intensity array.
    """
    foreground = _image.foreground_values(data, mask)
    # statistics in float64: float32 accumulation can make the std of
    # near-constant foregrounds spuriously nonzero
    foreground64 = foreground.astype(np.float64)
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
    normalized = zscore_array(data, _image.unwrap_mask(image, mask), norm_value=norm_value)
    return restore(normalized)
