"""Type-preserving extraction/restoration of image data (private).

This is the only module in the package that knows about nibabel. Every public
function routes through :func:`unwrap` so that numpy in -> numpy out and
nibabel in -> nibabel out (affine/header preserved), with the actual data
always handed to the math as float32 arrays.
"""

from __future__ import annotations

import dataclasses
import typing

import nibabel as nib
import nibabel.spatialimages  # explicit so nib.spatialimages resolves
import numpy as np

from intensity_normalization.errors import IntensityNormalizationError

__all__ = [
    "BinaryMask",
    "ForegroundIntensities",
    "Image",
    "ImageMeta",
    "IntensityArray",
    "Mask",
    "MaskArray",
    "foreground_values",
    "resolve_foreground",
    "restore",
    "unwrap",
    "unwrap_mask",
]

type AnyShape = tuple[int, ...]
type OneDimShape = tuple[int]

type IntensityArray = np.ndarray[AnyShape, np.dtype[np.floating]]
"""Image-shaped float data — the currency of every math function in the package."""

type ForegroundIntensities = np.ndarray[OneDimShape, np.dtype[np.floating]]
"""1-D samples of the intensities inside a foreground (brain) mask."""

type BinaryMask = np.ndarray[AnyShape, np.dtype[np.bool_]]
"""A thresholded mask: boolean array, True inside the region of interest."""

type MaskArray = IntensityArray | BinaryMask
"""A mask in array form: float (thresholded at > 0) or already-binary."""

type Image = IntensityArray | nib.spatialimages.SpatialImage
"""An MR image as users hand it to us: a plain intensity array or a nibabel spatial image."""

type Mask = MaskArray | nib.spatialimages.SpatialImage
"""A mask as users hand it to us: a float or bool array, or a nibabel image."""


@dataclasses.dataclass(frozen=True)
class ImageMeta:
    """Everything needed to rebuild the user's image from an array (a value,
    not a closure): the source class, its affine, and a copy of its header."""

    cls: type
    affine: np.ndarray | None
    header: typing.Any | None  # nibabel header, copied at unwrap time


_BACKGROUND_THRESHOLD = 1e-6


def unwrap(image: Image | BinaryMask) -> tuple[IntensityArray, ImageMeta]:
    """Split ``image`` into ``(data, meta)``: float32 array plus its context value.

    Use :func:`restore` to wrap an array of the same shape back into the
    source type (the header is copied at unwrap, so the source image's header
    is never mutated).
    """
    if isinstance(image, np.ndarray):
        return np.asarray(image, dtype=np.float32), ImageMeta(np.ndarray, None, None)

    if isinstance(image, nib.spatialimages.SpatialImage):
        meta = ImageMeta(type(image), image.affine, image.header.copy())
        return np.asanyarray(image.dataobj, dtype=np.float32), meta

    raise TypeError(f"Unsupported image type: {type(image)}. Pass a numpy array or a nibabel spatial image.")


def restore(meta: ImageMeta, data: IntensityArray) -> Image:
    """Wrap ``data`` back into the image type ``meta`` describes (pure).

    The header's datatype is set to the data's: without this, an int16 source
    header would silently truncate float32 normalized data on save. Any
    scaling is cleared so stored values equal the computed ones.
    """
    if meta.cls is np.ndarray:
        return np.asarray(data)
    assert meta.header is not None  # nibabel sources always carry a header copy
    header = meta.header.copy()
    header.set_data_dtype(np.asanyarray(data).dtype)
    if hasattr(header, "set_slope_inter"):
        header.set_slope_inter(None, None)  # nibabel headers are duck-typed; hasattr guards this
    return meta.cls(np.asanyarray(data), meta.affine, header)


def unwrap_mask(image: Image, mask: Mask | None) -> BinaryMask | None:
    """Unwrap ``mask``, binarize it (``> 0``), and validate it against ``image``.

    None passes through. For nibabel pairs the affines must agree: a mask in
    a different space with a matching shape would otherwise silently corrupt
    results. Shape equality itself is enforced in :func:`resolve_foreground`.
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
    return unwrap(mask)[0] > 0.0


def resolve_foreground(image: IntensityArray, mask: BinaryMask | None) -> BinaryMask:
    """The one owner of foreground semantics; no core ever sees ``mask=None``.

    ``mask=None`` is estimated as positive voxels (the only place that policy
    lives); a given mask is validated for shape and non-emptiness.

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
        out = mask
    if not out.any():
        msg = (
            "The foreground is empty: no positive voxels inside the mask. "
            "Check that the mask aligns with the image (same space and orientation)."
        )
        raise IntensityNormalizationError(msg)
    return out


def foreground_values(image: IntensityArray, foreground: BinaryMask) -> ForegroundIntensities:
    """1D array of the in-foreground intensities of ``image``."""
    return image[foreground]
