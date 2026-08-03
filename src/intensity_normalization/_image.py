"""Type-preserving extraction/restoration of image data (private).

This is the only module in the package that knows about nibabel. Every public
function routes through :func:`unwrap` so that numpy in -> numpy out and
nibabel in -> nibabel out (affine/header preserved), with the actual data
always handed to the math as float32 arrays.
"""

from __future__ import annotations

from collections.abc import Callable

import nibabel as nib
import numpy as np
import numpy.typing as npt

from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["ImageLike", "foreground_values", "get_mask", "unwrap"]

ImageLike = npt.NDArray[np.floating] | nib.spatialimages.SpatialImage
Restorer = Callable[[npt.NDArray[np.floating]], ImageLike]

_BACKGROUND_THRESHOLD = 1e-6


def unwrap(image: ImageLike, /) -> tuple[npt.NDArray[np.floating], Restorer]:
    """Return ``(data, restore)`` where ``restore`` wraps data back into image's type.

    ``data`` is a float32 array. ``restore`` must be called with an array of
    the same shape as ``data`` (or any shape for numpy inputs).
    """
    if isinstance(image, np.ndarray):
        return np.asarray(image, dtype=np.float32), np.asarray

    if isinstance(image, nib.spatialimages.SpatialImage):

        def restore_nibabel(data: npt.NDArray[np.floating]) -> ImageLike:
            return image.__class__(np.asanyarray(data), image.affine, image.header)

        return np.asanyarray(image.dataobj, dtype=np.float32), restore_nibabel

    raise TypeError(f"Unsupported image type: {type(image)}. Pass a numpy array or a nibabel spatial image.")


def get_mask(
    image: npt.NDArray[np.floating],
    /,
    mask: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.bool_]:
    """Boolean foreground mask; estimated as positive voxels when ``mask`` is None.

    Raises:
        IntensityNormalizationError: mask shape mismatch or empty foreground,
            with messages that say how to fix it.
    """
    if mask is None:
        if image.min() < 0.0:
            msg = (
                "The image contains negative values, so the foreground cannot "
                "be estimated as positive voxels. Provide a foreground (brain) mask."
            )
            raise IntensityNormalizationError(msg)
        out = image > _BACKGROUND_THRESHOLD
    else:
        if mask.shape != image.shape:
            msg = (
                f"Mask shape {mask.shape} does not match image shape {image.shape}. "
                "The mask must be resampled to the image space first."
            )
            raise IntensityNormalizationError(msg)
        out = mask > 0.0
    if not out.any():
        msg = (
            "The foreground is empty: no positive voxels inside the mask. "
            "Check that the mask aligns with the image (same space and orientation)."
        )
        raise IntensityNormalizationError(msg)
    return out


def foreground_values(
    image: npt.NDArray[np.floating],
    /,
    mask: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """1D array of the foreground (in-mask) intensities of ``image``."""
    return image[get_mask(image, mask)]
