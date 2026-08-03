"""Least-squares tissue mean normalization of a set of images.

Scales each image so its CSF/GM/WM tissue means match, in a least-squares
sense, the standard tissue means learned from a reference image.
"""

from __future__ import annotations

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


class LSQTransform(FittedTransform):
    """Least-squares scaling toward standard tissue means.

    Attributes are the learned parameters, exposed read-only.
    """

    method: typing.ClassVar[str] = "lsq"

    def __init__(
        self,
        standard_tissue_means: npt.NDArray[np.floating],
        *,
        norm_value: float = 1.0,
        seed: int | None = 0,
    ) -> None:
        self.standard_tissue_means = np.asarray(standard_tissue_means, dtype=np.float32)
        self.norm_value = norm_value
        self.seed = seed

    def _scale(
        self,
        data: npt.NDArray[np.floating],
        foreground_mask: npt.NDArray[np.bool_],
        membership: npt.NDArray[np.floating] | None,
    ) -> float:
        if membership is not None:
            if membership.shape[: data.ndim] != data.shape or membership.shape[-1] != 3:
                msg = (
                    f"Membership must have shape {(*data.shape, 3)}; got "
                    f"{membership.shape}. It must come from a co-registered T1-w image."
                )
                raise IntensityNormalizationError(msg)
            membership_map = np.asarray(membership, dtype=np.float32)
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
        mask_data = _image.unwrap(mask)[0] if mask is not None else None
        foreground_mask = _image.get_mask(data, mask_data)
        scale = self._scale(data, foreground_mask, membership)
        if scale == 0.0:
            msg = "Least-squares scale factor is zero; cannot normalize. Check the image and mask."
            raise IntensityNormalizationError(msg)
        return restore(data / scale * self.norm_value)

    def _state_dict(self) -> dict[str, np.ndarray]:
        return {
            "standard_tissue_means": self.standard_tissue_means,
            "norm_value": np.array(self.norm_value),
        }

    @classmethod
    def _from_state_dict(cls, state: dict[str, np.ndarray]) -> LSQTransform:
        return cls(state["standard_tissue_means"], norm_value=float(state["norm_value"]))


def fit(
    images: Sequence[ImageLike],
    /,
    masks: Sequence[ImageLike | None] | None = None,
    *,
    norm_value: float = 1.0,
    seed: int | None = 0,
    membership: npt.NDArray[np.floating] | None = None,
    return_tissue_maps: bool = False,
) -> LSQTransform | tuple[LSQTransform, npt.NDArray[np.floating]]:
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
        return_tissue_maps: also return the reference image's membership map
            (shape ``image.shape + (3,)``, CSF/GM/WM).

    Returns:
        A fitted :class:`LSQTransform` (and the tissue map if requested).
    """
    if len(images) == 0:
        raise IntensityNormalizationError("No images provided to fit.")
    if masks is not None and len(masks) != len(images):
        raise ValueError(f"Got {len(images)} images but {len(masks)} masks.")

    data, _ = _image.unwrap(images[0])
    mask = masks[0] if masks is not None else None
    mask_data = _image.unwrap(mask)[0] if mask is not None else None
    foreground_mask = _image.get_mask(data, mask_data)

    if membership is not None:
        if membership.shape[: data.ndim] != data.shape or membership.shape[-1] != 3:
            msg = (
                f"Membership must have shape {(*data.shape, 3)}; got "
                f"{membership.shape}. It must come from a co-registered T1-w image."
            )
            raise IntensityNormalizationError(msg)
        membership_map = np.asarray(membership, dtype=np.float32)
    else:
        _, membership_map = tissue_means(data, foreground_mask, seed=seed)
    foreground = data[foreground_mask]
    csf_mean = float(np.average(foreground, weights=membership_map[..., 0][foreground_mask]))
    if csf_mean == 0.0:
        msg = "The CSF mean of the reference image is zero. Check the image and mask."
        raise IntensityNormalizationError(msg)
    normed = data / csf_mean * norm_value
    standard_means, _ = tissue_means(normed, foreground_mask, seed=seed)

    tx = LSQTransform(standard_means, norm_value=norm_value, seed=seed)
    if return_tissue_maps:
        return tx, membership_map
    return tx


def fit_transform(
    images: Sequence[ImageLike],
    /,
    masks: Sequence[ImageLike | None] | None = None,
    **kwargs: typing.Any,
) -> tuple[LSQTransform, list[ImageLike]]:
    """Fit on the reference image and normalize all ``images``."""
    kwargs.pop("return_tissue_maps", None)  # fit_transform always returns images
    tx = typing.cast(LSQTransform, fit(images, masks, **kwargs))
    normed = [tx(img, masks[i] if masks is not None else None) for i, img in enumerate(images)]
    return tx, normed
