"""Least-squares tissue mean normalization of a set of images.

Scales each image so its CSF/GM/WM tissue means match, in a least-squares
sense, the standard tissue means learned from a reference image.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from intensity_normalization import _image
from intensity_normalization._image import ImageLike
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods._transform import FittedTransform
from intensity_normalization.methods.fcm import tissue_means

__all__ = ["LSQTransform", "fit", "fit_transform"]


def _check_membership(
    membership: npt.NDArray[np.floating] | None,
    data_shape: tuple[int, ...],
) -> npt.NDArray[np.floating] | None:
    """The one owner of the membership-map contract (single validation site)."""
    if membership is None:
        return None
    membership_map = np.asarray(membership, dtype=np.float32)
    if membership_map.shape != (*data_shape, 3):
        msg = (
            f"Membership must have shape {(*data_shape, 3)}; got "
            f"{membership_map.shape}. It must come from a co-registered T1-w image."
        )
        raise IntensityNormalizationError(msg)
    return membership_map


@dataclasses.dataclass(frozen=True, eq=False)
class LSQTransform(FittedTransform):
    """Least-squares scaling toward standard tissue means.

    ``reference_membership`` is the reference image's CSF/GM/WM membership map
    (shape ``image.shape + (3,)``), always computed during fitting — for
    diagnostics and tissue-map export, *not* for transforming new images (a
    new image is segmented from itself unless it is co-registered to the
    reference). Frozen and write-protected: the learned parameters cannot
    change after construction.
    """

    standard_tissue_means: npt.NDArray[np.floating]
    reference_membership: npt.NDArray[np.floating]
    norm_value: float = 1.0
    seed: int | None = 0

    method: typing.ClassVar[str] = "lsq"

    def __post_init__(self) -> None:
        object.__setattr__(self, "standard_tissue_means", np.asarray(self.standard_tissue_means, dtype=np.float32))
        object.__setattr__(self, "reference_membership", np.asarray(self.reference_membership, dtype=np.float32))
        self.standard_tissue_means.setflags(write=False)
        self.reference_membership.setflags(write=False)

    def _scale(
        self,
        data: npt.NDArray[np.floating],
        foreground_mask: npt.NDArray[np.bool_],
        membership: npt.NDArray[np.floating] | None,
    ) -> float:
        membership_map = _check_membership(membership, tuple(data.shape))
        if membership_map is not None:
            means = np.array(
                [
                    np.average(
                        data[foreground_mask],
                        weights=membership_map[..., i][foreground_mask],
                    )
                    for i in range(3)
                ]
            )
        else:
            means, _ = tissue_means(data, foreground_mask, seed=self.seed)
        numerator = float(means @ means)
        denominator = float(means @ self.standard_tissue_means)
        if denominator == 0.0:
            msg = (
                "Tissue means are orthogonal to the standard tissue means; "
                "cannot compute a least-squares scale. Is this a T1-w brain image?"
            )
            raise IntensityNormalizationError(msg)
        return numerator / denominator

    def transform(
        self,
        image: ImageLike,
        /,
        mask: ImageLike | None = None,
        *,
        membership: npt.NDArray[np.floating] | None = None,
    ) -> ImageLike:
        data, restore = _image.unwrap(image)
        mask_data = _image.unwrap_mask(image, mask)
        foreground_mask = _image.get_mask(data, mask_data)
        scale = self._scale(data, foreground_mask, membership)
        if scale == 0.0:
            msg = "Least-squares scale factor is zero; cannot normalize. Check the image and mask."
            raise IntensityNormalizationError(msg)
        return restore(data / scale * self.norm_value)

    def _state_dict(self) -> dict[str, np.ndarray]:
        return {
            "standard_tissue_means": self.standard_tissue_means,
            "reference_membership": self.reference_membership,
            "norm_value": np.array(self.norm_value),
            "seed": np.array(-1 if self.seed is None else self.seed),
        }

    @classmethod
    def _from_state_dict(cls, state: dict[str, np.ndarray]) -> LSQTransform:
        seed = int(state["seed"])
        return cls(
            state["standard_tissue_means"],
            state["reference_membership"],
            norm_value=float(state["norm_value"]),
            seed=None if seed < 0 else seed,
        )


def fit(
    images: Sequence[ImageLike],
    /,
    masks: Sequence[ImageLike | None] | None = None,
    *,
    norm_value: float = 1.0,
    seed: int | None = 0,
    membership: npt.NDArray[np.floating] | None = None,
) -> LSQTransform:
    """Learn standard tissue means from a reference image.

    The first image is the reference (per the original method): its tissue
    means, computed after scaling its CSF mean to ``norm_value``, become the
    standard that other images are scaled toward.

    Args:
        images: T1-w MR images (numpy arrays or nibabel images).
        masks: optional foreground (brain) mask per image.
        norm_value: intensity the reference CSF mean is mapped to.
        seed: RNG seed for the FCM tissue fit; ``None`` is nondeterministic.
        membership: precomputed membership map of the reference image (shape
            ``image.shape + (3,)``) for non-T1-w references; computed from the
            reference image itself when None.

    Returns:
        A fitted :class:`LSQTransform`. Its ``reference_membership`` attribute
        holds the reference image's CSF/GM/WM membership map.
    """
    if len(images) == 0:
        raise IntensityNormalizationError("No images provided to fit.")
    if masks is not None and len(masks) != len(images):
        raise ValueError(f"Got {len(images)} images but {len(masks)} masks.")

    data, _ = _image.unwrap(images[0])
    mask = masks[0] if masks is not None else None
    mask_data = _image.unwrap_mask(images[0], mask)
    foreground_mask = _image.get_mask(data, mask_data)

    membership_map = _check_membership(membership, tuple(data.shape))
    if membership_map is None:
        _, membership_map = tissue_means(data, foreground_mask, seed=seed)
    foreground = data[foreground_mask]
    csf_mean = float(np.average(foreground, weights=membership_map[..., 0][foreground_mask]))
    if csf_mean == 0.0:
        msg = "The CSF mean of the reference image is zero. Check the image and mask."
        raise IntensityNormalizationError(msg)
    normed = data / csf_mean * norm_value
    standard_means, _ = tissue_means(normed, foreground_mask, seed=seed)
    return LSQTransform(standard_means, membership_map, norm_value=norm_value, seed=seed)


def fit_transform(
    images: Sequence[ImageLike],
    /,
    masks: Sequence[ImageLike | None] | None = None,
    **kwargs: typing.Any,
) -> tuple[LSQTransform, list[ImageLike]]:
    """Fit on the reference image and normalize all ``images``."""
    tx = fit(images, masks, **kwargs)
    normed = [tx(img, masks[i] if masks is not None else None) for i, img in enumerate(images)]
    return tx, normed
