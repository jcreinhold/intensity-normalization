"""RAVEL normalization (WhiteStripe, then CSF control-voxel correction).

RAVEL (Fortin et al., 2017) is a *batch* correction method: it removes
technical variation by regressing out latent "unwanted factors" estimated from
the across-image variation of CSF control voxels. There is no single-image
transform — new images must be included in the batch. ``fit_transform`` is
therefore the entire API; the returned :class:`RavelResult` holds the learned
artifacts (factors, control mask) for diagnostics and reproducibility.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence
from os import PathLike

import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg

from intensity_normalization import _image
from intensity_normalization._image import ImageLike
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods.fcm import tissue_means
from intensity_normalization.methods.whitestripe import whitestripe

__all__ = ["RavelResult", "fit_transform"]


class RavelResult:
    """Artifacts learned by a RAVEL batch correction.

    Attributes are exposed read-only for diagnostics: the unwanted factors,
    the CSF control-voxel mask, and the control-voxel matrix.
    """

    method: typing.ClassVar[str] = "ravel"
    format_version: typing.ClassVar[int] = 1

    def __init__(
        self,
        unwanted_factors: npt.NDArray[np.floating],
        control_mask: npt.NDArray[np.bool_],
        control_voxels: npt.NDArray[np.floating],
        *,
        num_unwanted_factors: int,
    ) -> None:
        self.unwanted_factors = unwanted_factors
        self.control_mask = control_mask
        self.control_voxels = control_voxels
        self.num_unwanted_factors = num_unwanted_factors

    def save(self, path: str | PathLike[str], /) -> None:
        """Save the learned artifacts to ``path`` (``.npz``) for provenance."""
        np.savez(
            path,
            _method=np.array(self.method),
            _format_version=np.array(self.format_version),
            unwanted_factors=self.unwanted_factors,
            control_mask=self.control_mask,
            control_voxels=self.control_voxels,
            num_unwanted_factors=np.array(self.num_unwanted_factors),
        )

    @classmethod
    def load(cls, path: str | PathLike[str], /) -> RavelResult:
        """Load artifacts saved with :meth:`save`."""
        with np.load(path) as data:
            state = {k: data[k] for k in data.files}
        method = str(state.pop("_method"))
        version = int(state.pop("_format_version"))
        if method != cls.method:
            raise ValueError(f"{path} holds {method!r} artifacts, not 'ravel'.")
        if version != cls.format_version:
            raise ValueError(
                f"{path} uses ravel format version {version}; this version reads version {cls.format_version}."
            )
        return cls(
            state["unwanted_factors"],
            state["control_mask"].astype(bool),
            state["control_voxels"],
            num_unwanted_factors=int(state["num_unwanted_factors"]),
        )


def _unwanted_factors(
    control_voxels: npt.NDArray[np.floating],
    num_unwanted_factors: int,
    sparse_svd: bool,
) -> npt.NDArray[np.floating]:
    if sparse_svd:
        _, _, vh = scipy.sparse.linalg.svds(
            scipy.sparse.bsr_matrix(control_voxels),
            k=num_unwanted_factors,
            return_singular_vectors="vh",
        )
    else:
        _, _, vh = np.linalg.svd(control_voxels, full_matrices=False)
    return vh.T[:, :num_unwanted_factors]


def _correction(
    image_matrix: npt.NDArray[np.floating],
    unwanted_factors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Remove the unwanted-factor trend from each voxel's across-image course."""
    beta = np.linalg.solve(
        unwanted_factors.T @ unwanted_factors,
        unwanted_factors.T @ image_matrix.T,
    )
    fitted = (unwanted_factors @ beta).T
    residuals = image_matrix - fitted
    voxel_means = image_matrix.mean(axis=1, keepdims=True)
    return residuals + voxel_means


def fit_transform(
    images: Sequence[ImageLike],
    /,
    masks: Sequence[ImageLike | None] | None = None,
    *,
    register: bool = True,
    membership_threshold: float = 0.99,
    num_unwanted_factors: int = 1,
    sparse_svd: bool = False,
    quantile_to_label_csf: float = 1.0,
    masks_are_csf: bool = False,
    template: ImageLike | None = None,
    whitestripe_kwargs: dict[str, typing.Any] | None = None,
    seed: int | None = 0,
) -> tuple[RavelResult, list[ImageLike]]:
    """WhiteStripe-normalize then RAVEL-correct a set of co-registered images.

    All images must have the same shape and be (at least rigidly) co-registered;
    good results require deformable co-registration. With ``register=True``
    (default), images are deformably registered to a template (the first image
    unless ``template`` is given), the correction is computed in template
    space, and the corrected images are warped back to native space.

    Args:
        images: MR images (numpy arrays or nibabel images), all one modality.
        masks: foreground (brain) mask per image (CSF masks if
            ``masks_are_csf``).
        register: deformably register to a template before finding control
            voxels. Requires antspy. If False, images are assumed already
            deformably co-registered.
        membership_threshold: FCM CSF membership threshold for control voxels.
        num_unwanted_factors: ``b`` in the RAVEL paper.
        sparse_svd: use a sparse SVD (lower memory) for the factor estimation.
        quantile_to_label_csf: fraction of images in which a voxel must be CSF
            to be a control voxel (1.0 = strict intersection).
        masks_are_csf: ``masks`` are boolean CSF masks, not brain masks
            (implies ``register=False``).
        template: registration target; defaults to the first image.
        whitestripe_kwargs: extra kwargs for the WhiteStripe step.
        seed: RNG seed for the FCM tissue fits; ``None`` is nondeterministic.

    Returns:
        ``(result, normalized)``: the learned :class:`RavelResult` artifacts
        and the normalized images (same types as the inputs).
    """
    if len(images) == 0:
        raise IntensityNormalizationError("No images provided to normalize.")
    if masks is not None and len(masks) != len(images):
        raise ValueError(f"Got {len(images)} images but {len(masks)} masks.")
    if register and masks_are_csf:
        raise ValueError("masks_are_csf implies the images are already co-registered; set register=False.")

    ws_kwargs = dict(whitestripe_kwargs or {})
    ws_kwargs.setdefault("seed", seed)

    datas = [_image.unwrap(img)[0] for img in images]
    mask_datas = [_image.unwrap(m)[0] if m is not None else None for m in masks] if masks else [None] * len(images)
    shape = datas[0].shape
    for i, data in enumerate(datas):
        if data.shape != shape:
            msg = (
                f"All images must have the same shape and voxel-wise correspondence; "
                f"image 0 is {shape} but image {i} is {data.shape}. Co-register the "
                f"images first (see intensity_normalization.coregister)."
            )
            raise IntensityNormalizationError(msg)

    n_images = len(images)
    image_matrix = np.zeros((int(np.prod(shape)), n_images), dtype=np.float32)
    csf_masks: list[npt.NDArray[np.bool_]] = []

    # WhiteStripe each image, in registration space when registering
    if register:
        from intensity_normalization.tools.ants import require_ants, to_ants

        ants = require_ants()
        fixed = to_ants(template) if template is not None else to_ants(datas[0])
        inverse_transforms: list[list[str]] = []
        natives = [to_ants(d) for d in datas]
        for i, (data, mask_data) in enumerate(zip(datas, mask_datas, strict=True)):
            ws_data = _image.unwrap(whitestripe(data, mask_data, **ws_kwargs))[0]
            registration = ants.registration(
                fixed=fixed,
                moving=natives[i],
                type_of_transform="SyN",
                aff_metric="mattes",
                syn_metric="mattes",
            )
            ws_ants = to_ants(ws_data)
            registered = ants.apply_transforms(fixed, ws_ants, registration["fwdtransforms"])
            inverse_transforms.append(registration["invtransforms"])
            ws_reg = registered.numpy().astype(np.float32)
            image_matrix[:, i] = ws_reg.ravel()
            csf_masks.append(_csf_mask(ws_reg, mask_data, masks_are_csf, membership_threshold, seed, i))
        work_shape = fixed.shape
    else:
        inverse_transforms = []
        natives = []
        for i, (data, mask_data) in enumerate(zip(datas, mask_datas, strict=True)):
            ws_data = _image.unwrap(whitestripe(data, mask_data, **ws_kwargs))[0]
            image_matrix[:, i] = ws_data.ravel()
            csf_masks.append(_csf_mask(ws_data, mask_data, masks_are_csf, membership_threshold, seed, i))
        work_shape = shape

    control_mask_sum = np.stack([csf.ravel() for csf in csf_masks]).sum(axis=0)
    threshold = np.floor(n_images * quantile_to_label_csf)
    control_mask = (control_mask_sum >= threshold).reshape(work_shape)
    if not control_mask.any():
        msg = (
            "No common CSF control voxels found across the image set. "
            "Lower membership_threshold or quantile_to_label_csf, or check "
            "that the images are co-registered."
        )
        raise IntensityNormalizationError(msg)
    control_voxels = image_matrix[control_mask.ravel(), :]

    unwanted = _unwanted_factors(control_voxels, num_unwanted_factors, sparse_svd)
    normalized_matrix = _correction(image_matrix, unwanted)

    normalized: list[ImageLike] = []
    for i, image in enumerate(images):
        _, restore = _image.unwrap(image)
        corrected = normalized_matrix[:, i].reshape(work_shape)
        if register:
            corrected_ants = ants.apply_transforms(natives[i], to_ants(corrected), inverse_transforms[i])
            corrected = corrected_ants.numpy().astype(np.float32)
        normalized.append(restore(corrected.astype(np.float32)))

    result = RavelResult(
        unwanted,
        control_mask,
        control_voxels,
        num_unwanted_factors=num_unwanted_factors,
    )
    return result, normalized


def _csf_mask(
    ws_data: npt.NDArray[np.floating],
    mask_data: npt.NDArray[np.floating] | None,
    masks_are_csf: bool,
    membership_threshold: float,
    seed: int | None,
    index: int,
) -> npt.NDArray[np.bool_]:
    """Boolean CSF control mask for one (WhiteStripe-normalized) image."""
    if masks_are_csf:
        if mask_data is None:
            raise ValueError("masks_are_csf=True requires CSF masks in `masks`.")
        return mask_data > 0
    foreground_mask = _image.get_mask(ws_data, mask_data)
    _, membership_map = tissue_means(ws_data, foreground_mask, seed=seed)
    csf = membership_map[..., 0] > membership_threshold
    if not csf.any():
        msg = (
            f"No CSF control voxels found in image {index} at membership threshold "
            f"{membership_threshold}. Lower the threshold or check the image."
        )
        raise IntensityNormalizationError(msg)
    return csf
