"""Type-preserving extraction/restoration of image data (private).

This is the only module in the package that knows about nibabel. Every public
function routes through :func:`unwrap` so that numpy in -> numpy out and
nibabel in -> nibabel out (affine/header preserved), with the actual data
always handed to the math as float32 arrays.
"""

from __future__ import annotations

from collections.abc import Callable

import nibabel as nib
import nibabel.spatialimages  # explicit so nib.spatialimages resolves
import numpy as np

from intensity_normalization.errors import IntensityNormalizationError

__all__ = [
    "ForegroundIntensities",
    "Image",
    "IntensityArray",
    "Mask",
    "MaskArray",
    "foreground_values",
    "get_mask",
    "unwrap",
    "unwrap_mask",
]

type AnyShape = tuple[int, ...]
type OneDimShape = tuple[int]

type IntensityArray = np.ndarray[AnyShape, np.dtype[np.floating]]
"""Image-shaped float data — the currency of every math function in the package."""

type ForegroundIntensities = np.ndarray[OneDimShape, np.dtype[np.floating]]
"""1-D samples of the intensities inside a foreground (brain) mask."""

type MaskArray = np.ndarray[AnyShape, np.dtype[np.bool_]]
"""Boolean mask array, True inside the region of interest."""

type Image = IntensityArray | nib.spatialimages.SpatialImage
"""An MR image as users hand it to us: a plain intensity array or a nibabel spatial image."""

type Mask = IntensityArray | MaskArray | nib.spatialimages.SpatialImage
"""A mask as users hand it to us: a float or bool array, or a nibabel image."""

type Restorer = Callable[[IntensityArray], Image]

_BACKGROUND_THRESHOLD = 1e-6


def unwrap(image: Image | MaskArray) -> tuple[IntensityArray, Restorer]:
    """Return ``(data, restore)`` where ``restore`` wraps data back into image's type.

    ``data`` is a float32 array. ``restore`` must be called with an array of
    the same shape as ``data`` (or any shape for numpy inputs).
    """
    if isinstance(image, np.ndarray):
        return np.asarray(image, dtype=np.float32), np.asarray

    if isinstance(image, nib.spatialimages.SpatialImage):

        def restore_nibabel(data: IntensityArray) -> Image:
            # Copy the header so the source image is never mutated, and set its
            # datatype to the data's: without this, an int16 source header would
            # silently truncate float32 normalized data on save. Clear any
            # scaling so stored values equal the computed ones.
            header = image.header.copy()
            header.set_data_dtype(np.asanyarray(data).dtype)
            if hasattr(header, "set_slope_inter"):
                header.set_slope_inter(None, None)  # ty: ignore[call-non-callable]  # nibabel headers are duck-typed; hasattr guards this
            return image.__class__(np.asanyarray(data), image.affine, header)

        return np.asanyarray(image.dataobj, dtype=np.float32), restore_nibabel

    raise TypeError(f"Unsupported image type: {type(image)}. Pass a numpy array or a nibabel spatial image.")


def unwrap_mask(image: Image, mask: Mask | None) -> IntensityArray | None:
    """Unwrap ``mask`` and validate it against ``image`` (None passes through).

    For nibabel pairs the affines must agree: a mask in a different space with
    a matching shape would otherwise silently corrupt results. Shape equality
    itself is enforced in :func:`get_mask`.
    """
    if mask is None:
        return None
    if (
        isinstance(image, nib.spatialimages.SpatialImage)
        and isinstance(mask, nib.spatialimages.SpatialImage)
        and not np.allclose(image.affine, mask.affine, rtol=0.0, atol=1e-3)
    ):
        msg = (
            "The mask and image are in different spaces (affines differ). "
            "Resample the mask into the image's space before normalizing."
        )
        raise IntensityNormalizationError(msg)
    return unwrap(mask)[0]


def get_mask(image: IntensityArray, mask: IntensityArray | MaskArray | None = None) -> MaskArray:
    """Boolean foreground mask; estimated as positive voxels when ``mask`` is None.

    Raises:
        IntensityNormalizationError: mask shape mismatch or empty foreground,
            with messages that say how to fix it.
    """
    if mask is None:
        if np.min(image) < 0.0:
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


def foreground_values(image: IntensityArray, mask: IntensityArray | None = None) -> ForegroundIntensities:
    """1D array of the foreground (in-mask) intensities of ``image``."""
    return image[get_mask(image, mask)]
