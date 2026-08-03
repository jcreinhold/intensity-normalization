"""ANTs-based tools: co-registration and preprocessing (N4 + resample/reorient).

ANTsPy is an optional dependency (``pip install intensity-normalization[ants]``),
imported lazily inside functions so the base install never touches it.
"""

from __future__ import annotations

import typing

import nibabel as nib
import numpy as np

from intensity_normalization._ants import require_ants, to_ants
from intensity_normalization._image import ImageLike

__all__ = ["coregister", "preprocess", "require_ants", "to_ants"]

if typing.TYPE_CHECKING:
    from ants.core.ants_image import ANTsImage


def _ants_to_nibabel(
    result: ANTsImage,
    reference: nib.spatialimages.SpatialImage,
) -> nib.spatialimages.SpatialImage:
    """Convert an ANTs result back to nibabel using the reference's affine/header."""
    data = result.numpy().astype(np.float32)
    if data.shape == reference.shape:
        return reference.__class__(data, reference.affine, reference.header)
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(result.direction) * np.asarray(result.spacing)
    affine[:3, 3] = np.asarray(result.origin)
    return nib.nifti1.Nifti1Image(data, affine)


def _from_ants(reference: ImageLike | ANTsImage, /, result: ANTsImage) -> ImageLike | ANTsImage:
    """Convert an ANTs result back to the type of the input it came from."""
    from ants.core.ants_image import ANTsImage as _ANTsImage

    if isinstance(reference, _ANTsImage):
        return result
    if isinstance(reference, nib.spatialimages.SpatialImage):
        return _ants_to_nibabel(result, reference)
    return result.numpy().astype(np.float32)


def coregister(
    image: ImageLike | ANTsImage,
    /,
    template: ImageLike | ANTsImage | None = None,
    *,
    type_of_transform: str = "Affine",
    interpolator: str = "bSpline",
    metric: str = "mattes",
    initial_rigid: bool = True,
    template_mask: ImageLike | ANTsImage | None = None,
) -> ImageLike | ANTsImage:
    """Register ``image`` to ``template`` with ANTs (MNI template if None).

    Args:
        image: moving image; the same type is returned.
        template: fixed image. If None, uses the MNI template bundled with ANTs.
        type_of_transform: ANTs transform type (e.g., "Rigid", "Affine", "SyN").
        interpolator: interpolation for the resampled output.
        metric: registration metric (e.g., "mattes", "CC").
        initial_rigid: do a rigid registration first to initialize.
        template_mask: mask restricting the metric on the fixed image.

    Returns:
        The registered image, same type as ``image``.
    """
    ants = require_ants()
    if template is None:
        template = ants.image_read(ants.get_ants_data("mni"))
    template_ants = to_ants(template)
    image_ants = to_ants(image)

    rigid_transform = None
    if initial_rigid:
        rigid = ants.registration(
            fixed=template_ants,
            moving=image_ants,
            type_of_transform="Rigid",
            aff_metric=metric,
            syn_metric=metric,
        )
        rigid_transform = rigid["fwdtransforms"][0]

    registration = ants.registration(
        fixed=template_ants,
        moving=image_ants,
        initial_transform=rigid_transform,
        type_of_transform=type_of_transform,
        mask=to_ants(template_mask) if template_mask is not None else None,
        aff_metric=metric,
        syn_metric=metric,
    )
    registered = ants.apply_transforms(
        template_ants,
        image_ants,
        registration["fwdtransforms"],
        interpolator=interpolator,
    )
    return _from_ants(image, registered)


_INTERP_TYPES: dict[str, int] = {
    "linear": 0,
    "nearest_neighbor": 1,
    "gaussian": 4,
    "windowed_sinc": 5,
    "bspline": 3,
}


def preprocess(
    image: ImageLike,
    /,
    mask: ImageLike | None = None,
    *,
    resolution: tuple[float, float, float] | None = None,
    orientation: str = "RAS",
    n4_convergence_options: dict[str, typing.Any] | None = None,
    interp_type: str = "linear",
    second_n4_with_smoothed_mask: bool = True,
) -> tuple[ImageLike, ImageLike]:
    """Preprocess an MR image: N4 bias correction, optional resample, reorientation.

    Args:
        image: numpy array or nibabel image; the same types are returned.
        mask: foreground (brain) mask; estimated from the image if None.
        resolution: voxel size (mm) to resample to; None skips resampling.
        orientation: ANTs orientation code (e.g., "RAS").
        n4_convergence_options: ANTs N4 convergence dict.
        interp_type: resampling interpolation ("linear", "nearest_neighbor",
            "gaussian", "windowed_sinc", "bspline").
        second_n4_with_smoothed_mask: run a second N4 weighted by a smoothed
            mask; usually improves the correction.

    Returns:
        ``(preprocessed_image, foreground_mask)``, same types as the inputs.
    """
    ants = require_ants()
    if n4_convergence_options is None:
        n4_convergence_options = {"iters": [200, 200, 200, 200], "tol": 1e-7}
    if interp_type not in _INTERP_TYPES:
        raise ValueError(f"Unknown interp_type {interp_type!r}. Choose one of: {', '.join(_INTERP_TYPES)}.")

    ants_image = to_ants(image)
    ants_mask = to_ants(mask) if mask is not None else ants_image.get_mask()

    ants_image = ants.n4_bias_field_correction(ants_image, convergence=n4_convergence_options)
    if second_n4_with_smoothed_mask:
        smoothed_mask = ants.smooth_image(ants_mask, 1.0)
        ants_image = ants.n4_bias_field_correction(
            ants_image,
            convergence=n4_convergence_options,
            weight_mask=smoothed_mask,
        )

    if resolution is not None and tuple(resolution) != tuple(ants_image.spacing):
        ants_mask = ants.resample_image(
            ants_mask, resolution, use_voxels=False, interp_type=_INTERP_TYPES["nearest_neighbor"]
        )
        ants_image = ants.resample_image(
            ants_image, resolution, use_voxels=False, interp_type=_INTERP_TYPES[interp_type]
        )

    if orientation is not None and ants_image.orientation != orientation:
        ants_image = ants_image.reorient_image2(orientation)
        ants_mask = ants_mask.reorient_image2(orientation)

    out_mask = _from_ants(mask if mask is not None else image, ants_mask > 0.5)
    out_image = _from_ants(image, ants_image * ants_mask)
    return out_image, out_mask
