"""RAVEL normalization (WhiteStripe, then CSF control-voxel correction).

RAVEL (Fortin et al., 2017) is a *batch* correction method: it removes
technical variation by regressing out latent "unwanted factors" estimated from
the across-image variation of CSF control voxels. There is no single-image
transform — new images must be included in the batch. ``fit_transform`` is
therefore the entire API; the returned :class:`RavelResult` holds the learned
artifacts (factors, control mask) for diagnostics and reproducibility.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable, Sequence
from os import PathLike

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from intensity_normalization import _image
from intensity_normalization._image import Image, IntensityArray, Mask, MaskArray
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods._transform import _load_stamped, _save_stamped
from intensity_normalization.methods.fcm import tissue_means
from intensity_normalization.methods.whitestripe import whitestripe_array

__all__ = ["RavelResult", "fit_transform", "ravel_array"]


@dataclasses.dataclass(frozen=True, eq=False)
class RavelResult:
    """Artifacts learned by a RAVEL batch correction.

    Attributes are exposed read-only for diagnostics: the unwanted factors,
    the CSF control-voxel mask, and the control-voxel matrix. These live in
    the *working space* — template space when ``register=True`` (the default),
    native space otherwise. Frozen and write-protected: the artifacts cannot
    change after construction, so a saved result always matches memory.
    """

    unwanted_factors: IntensityArray
    control_mask: MaskArray
    control_voxels: IntensityArray
    num_unwanted_factors: int = 1

    method: typing.ClassVar[str] = "ravel"
    format_version: typing.ClassVar[int] = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "unwanted_factors", np.asarray(self.unwanted_factors, dtype=np.float32))
        object.__setattr__(self, "control_mask", np.asarray(self.control_mask, dtype=bool))
        object.__setattr__(self, "control_voxels", np.asarray(self.control_voxels, dtype=np.float32))
        self.unwanted_factors.setflags(write=False)
        self.control_mask.setflags(write=False)
        self.control_voxels.setflags(write=False)

    def save(self, path: str | PathLike[str]) -> None:
        """Save the learned artifacts to ``path`` (``.npz``) for provenance."""
        _save_stamped(
            path,
            self.method,
            self.format_version,
            {
                "unwanted_factors": self.unwanted_factors,
                "control_mask": self.control_mask,
                "control_voxels": self.control_voxels,
                "num_unwanted_factors": np.array(self.num_unwanted_factors),
            },
        )

    @classmethod
    def load(cls, path: str | PathLike[str]) -> RavelResult:
        """Load artifacts saved with :meth:`save`."""
        state = _load_stamped(path, cls.method, cls.format_version)
        return cls(
            state["unwanted_factors"],
            state["control_mask"],
            state["control_voxels"],
            num_unwanted_factors=int(state["num_unwanted_factors"]),
        )


class _WorkSpace:
    """The space the across-image matrix is built in: native or template.

    Owns every piece of working-space knowledge so the rest of the pipeline
    cannot mix spaces by construction: the WhiteStriped images, the masks
    (warped into this space, never the caller's native-space masks), and how
    to move a corrected image back to image ``i``'s native space (identity in
    native space; the inverse registration warp in template space).
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        ws_images: list[IntensityArray],
        masks: list[IntensityArray | None],
        warp_backs: list[Callable[[IntensityArray], IntensityArray]] | None = None,
    ) -> None:
        self.shape = shape
        self.ws_images = ws_images
        self.masks = masks
        self._warp_backs = warp_backs

    def warp_back(self, index: int, corrected: IntensityArray) -> IntensityArray:
        """Move a corrected image from the working space to native space."""
        if self._warp_backs is None:
            return corrected
        return self._warp_backs[index](corrected)


def _native_space(
    ws_datas: list[IntensityArray],
    mask_datas: list[IntensityArray | None],
) -> _WorkSpace:
    return _WorkSpace(tuple(ws_datas[0].shape), ws_datas, mask_datas)


def _template_space(
    images: Sequence[Image],
    ws_datas: list[IntensityArray],
    mask_datas: list[IntensityArray | None],
    template: Image | None,
) -> _WorkSpace:
    """Register to a template; correction happens there, results warp back.

    Registration runs on the *original* images (better contrast than the
    WhiteStriped ones); the learned warps are applied to the WhiteStriped
    data and, with nearest-neighbor interpolation, to the masks — so the
    masks the pipeline sees always index working-space voxels, never
    native-space ones. ``ants.new_image_like`` carries the source geometry
    onto the numpy data so ANTs never sees identity spacing.
    """
    from intensity_normalization._ants import require_ants, to_ants

    ants = require_ants()
    fixed = to_ants(template) if template is not None else to_ants(images[0])
    ws_in_space: list[IntensityArray] = []
    masks_in_space: list[IntensityArray | None] = []
    warp_backs: list[Callable[[IntensityArray], IntensityArray]] = []
    for image, ws_data, mask_data in zip(images, ws_datas, mask_datas, strict=True):
        native = to_ants(image)
        registration = ants.registration(
            fixed=fixed,
            moving=native,
            type_of_transform="SyN",
            aff_metric="mattes",
            syn_metric="mattes",
        )
        warped = ants.apply_transforms(
            fixed,
            ants.new_image_like(native, ws_data),
            registration["fwdtransforms"],
        )
        ws_in_space.append(warped.numpy().astype(np.float32))
        if mask_data is not None:
            warped_mask = ants.apply_transforms(
                fixed,
                ants.new_image_like(native, mask_data),
                registration["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            masks_in_space.append(warped_mask.numpy().astype(np.float32))
        else:
            masks_in_space.append(None)

        def warp_back(
            corrected: IntensityArray,
            _native: typing.Any = native,
            _inv: list[str] = registration["invtransforms"],
        ) -> IntensityArray:
            back = ants.apply_transforms(_native, ants.new_image_like(fixed, corrected), _inv)
            return back.numpy().astype(np.float32)

        warp_backs.append(warp_back)
    return _WorkSpace(tuple(fixed.shape), ws_in_space, masks_in_space, warp_backs)


def _control_voxels(
    image_matrix: IntensityArray,
    csf_masks: Sequence[MaskArray],
    quantile_to_label_csf: float,
    shape: tuple[int, ...],
) -> tuple[MaskArray, IntensityArray]:
    """Control voxels: CSF in at least ``quantile_to_label_csf`` of the images."""
    count = np.stack([csf.ravel() for csf in csf_masks]).sum(axis=0)
    threshold = np.floor(image_matrix.shape[1] * quantile_to_label_csf)
    control_mask = (count >= threshold).reshape(shape)
    if not control_mask.any():
        msg = (
            "No common CSF control voxels found across the image set. "
            "Lower membership_threshold or quantile_to_label_csf, or check "
            "that the images are co-registered."
        )
        raise IntensityNormalizationError(msg)
    return control_mask, image_matrix[control_mask.ravel(), :]


def _unwanted_factors(
    control_voxels: IntensityArray,
    num_unwanted_factors: int,
    sparse_svd: bool,
) -> IntensityArray:
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
    image_matrix: IntensityArray,
    unwanted_factors: IntensityArray,
) -> IntensityArray:
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
    images: Sequence[Image],
    masks: Sequence[Mask | None] | None = None,
    *,
    register: bool = True,
    membership_threshold: float = 0.99,
    num_unwanted_factors: int = 1,
    sparse_svd: bool = False,
    quantile_to_label_csf: float = 1.0,
    masks_are_csf: bool = False,
    template: Image | None = None,
    whitestripe_kwargs: dict[str, typing.Any] | None = None,
    seed: int | None = 0,
) -> tuple[RavelResult, list[Image]]:
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
            voxels (masks are warped along). Requires antspy. If False, images
            are assumed already deformably co-registered.
        membership_threshold: FCM CSF membership threshold for control voxels.
        num_unwanted_factors: ``b`` in the RAVEL paper.
        sparse_svd: use a sparse SVD (lower memory) for the factor estimation.
        quantile_to_label_csf: fraction of images in which a voxel must be CSF
            to be a control voxel (1.0 = strict intersection).
        masks_are_csf: ``masks`` are boolean CSF masks, not brain masks.
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

    datas = [_image.unwrap(img)[0] for img in images]
    mask_datas: list[IntensityArray | None] = (
        [_image.unwrap_mask(img, m) for img, m in zip(images, masks, strict=True)] if masks else [None] * len(images)
    )
    shape = datas[0].shape
    for i, data in enumerate(datas):
        if data.shape != shape:
            msg = (
                f"All images must have the same shape and voxel-wise correspondence; "
                f"image 0 is {shape} but image {i} is {data.shape}. Co-register the "
                f"images first (see intensity_normalization.coregister)."
            )
            raise IntensityNormalizationError(msg)

    ws_kwargs = dict(whitestripe_kwargs or {})
    ws_kwargs.setdefault("seed", seed)
    ws_datas = [
        whitestripe_array(data, mask_data, **ws_kwargs) for data, mask_data in zip(datas, mask_datas, strict=True)
    ]

    space = _template_space(images, ws_datas, mask_datas, template) if register else _native_space(ws_datas, mask_datas)

    result, corrected_datas = ravel_array(
        space.ws_images,
        space.masks,
        membership_threshold=membership_threshold,
        num_unwanted_factors=num_unwanted_factors,
        sparse_svd=sparse_svd,
        quantile_to_label_csf=quantile_to_label_csf,
        masks_are_csf=masks_are_csf,
        seed=seed,
    )

    normalized: list[Image] = []
    for i, image in enumerate(images):
        _, restore = _image.unwrap(image)
        corrected = space.warp_back(i, corrected_datas[i])
        normalized.append(restore(corrected.astype(np.float32)))
    return result, normalized


def ravel_array(
    ws_images: Sequence[IntensityArray],
    masks: Sequence[IntensityArray | None] | None = None,
    *,
    membership_threshold: float = 0.99,
    num_unwanted_factors: int = 1,
    sparse_svd: bool = False,
    quantile_to_label_csf: float = 1.0,
    masks_are_csf: bool = False,
    seed: int | None = 0,
) -> tuple[RavelResult, list[IntensityArray]]:
    """RAVEL-correct a set of WhiteStripe-normalized, co-registered intensity arrays.

    This is the registration-free core: every array must already be in one
    common (working) space with voxel-wise correspondence.

    Args:
        ws_images: WhiteStripe-normalized intensity arrays, all the same shape.
        masks: foreground (brain) mask array per array (CSF masks if
            ``masks_are_csf``).
        membership_threshold: FCM CSF membership threshold for control voxels.
        num_unwanted_factors: ``b`` in the RAVEL paper.
        sparse_svd: use a sparse SVD (lower memory) for the factor estimation.
        quantile_to_label_csf: fraction of arrays in which a voxel must be CSF
            to be a control voxel (1.0 = strict intersection).
        masks_are_csf: ``masks`` are boolean CSF masks, not brain masks.
        seed: RNG seed for the FCM tissue fits; ``None`` is nondeterministic.

    Returns:
        ``(result, corrected)``: the learned :class:`RavelResult` artifacts
        and the corrected intensity arrays, all in the working space.
    """
    if len(ws_images) == 0:
        raise IntensityNormalizationError("No images provided to normalize.")
    if masks is not None and len(masks) != len(ws_images):
        raise ValueError(f"Got {len(ws_images)} images but {len(masks)} masks.")
    shape = tuple(ws_images[0].shape)

    image_matrix = np.stack([ws.ravel() for ws in ws_images], axis=1).astype(np.float32)
    csf_masks = [
        _csf_mask(ws, mask_data, masks_are_csf, membership_threshold, seed, i)
        for i, (ws, mask_data) in enumerate(zip(ws_images, masks or [None] * len(ws_images), strict=True))
    ]
    control_mask, control_voxels = _control_voxels(image_matrix, csf_masks, quantile_to_label_csf, shape)

    unwanted = _unwanted_factors(control_voxels, num_unwanted_factors, sparse_svd)
    normalized_matrix = _correction(image_matrix, unwanted)

    corrected = [normalized_matrix[:, i].reshape(shape) for i in range(len(ws_images))]
    result = RavelResult(
        unwanted,
        control_mask,
        control_voxels,
        num_unwanted_factors=num_unwanted_factors,
    )
    return result, corrected


def _csf_mask(
    ws_data: IntensityArray,
    mask_data: IntensityArray | None,
    masks_are_csf: bool,
    membership_threshold: float,
    seed: int | None,
    index: int,
) -> MaskArray:
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
