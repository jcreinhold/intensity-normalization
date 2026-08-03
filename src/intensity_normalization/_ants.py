"""Private bridge to ANTsPy: lazy import and numpy/nibabel → ANTs conversion.

Infrastructure shared by :mod:`intensity_normalization.methods.ravel` and
:mod:`intensity_normalization.tools.ants`. ANTsPy is an optional dependency
(``pip install intensity-normalization[ants]``), so the import happens only
inside :func:`require_ants`; the base install never touches it.
"""

from __future__ import annotations

import typing

import nibabel as nib
import nibabel.spatialimages  # explicit so nib.spatialimages resolves
import numpy as np

from intensity_normalization._image import Image
from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["require_ants", "to_ants"]

if typing.TYPE_CHECKING:
    from ants.core.ants_image import ANTsImage  # ty: ignore[unresolved-import]  # optional dep


def require_ants() -> typing.Any:
    """Import antspy or raise an actionable error."""
    try:
        import ants  # ty: ignore[unresolved-import]  # optional dep
    except ImportError as exn:
        msg = "This feature requires ANTsPy. Install it with: pip install 'intensity-normalization[ants]'"
        raise IntensityNormalizationError(msg) from exn
    return ants


def _nibabel_to_ants(image: nib.spatialimages.SpatialImage) -> ANTsImage:
    """Convert a nibabel image to ANTs, preserving origin/spacing/direction."""
    ants = require_ants()
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    affine = image.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    direction = affine[:3, :3] / spacing
    return ants.from_numpy(
        data,
        origin=tuple(affine[:3, 3]),
        spacing=tuple(spacing),
        direction=direction,
    )


def to_ants(image: Image | ANTsImage) -> ANTsImage:
    """Convert an image to an :class:`ants.ANTsImage`, preserving geometry."""
    ants = require_ants()
    if isinstance(image, ants.core.ants_image.ANTsImage):
        return image
    if isinstance(image, nib.spatialimages.SpatialImage):
        return _nibabel_to_ants(image)
    return ants.from_numpy(np.asarray(image, dtype=np.float32))
