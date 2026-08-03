"""FCM-based normalization: scale a tissue's fuzzy mean intensity to a fixed value."""

from __future__ import annotations

import numpy as np

from intensity_normalization import _image
from intensity_normalization._image import BinaryMask, Image, IntensityArray, Mask
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods import _fcm
from intensity_normalization.methods._common import standardize

__all__ = ["TISSUES", "fcm", "fcm_array", "tissue_means"]

#: Tissue classes in ascending T1-w intensity order (CSF < GM < WM).
TISSUES: tuple[str, ...] = ("csf", "gm", "wm")


def _tissue_index(tissue: str) -> int:
    tissue = tissue.lower()
    if tissue not in TISSUES:
        raise ValueError(f"Unknown tissue {tissue!r}. Choose one of: {', '.join(TISSUES)}.")
    return TISSUES.index(tissue)


def tissue_means(
    data: IntensityArray,
    foreground_mask: BinaryMask,
    *,
    seed: int | None = 0,
    max_samples: int = 200_000,
) -> tuple[IntensityArray, IntensityArray]:
    """Weighted tissue means (CSF, GM, WM) of an image via fuzzy c-means.

    Centers are fit on a seeded subsample of the foreground; memberships are
    then computed for every foreground voxel, so results are statistically
    identical to a full fit but much faster on large images.

    Returns:
        ``(means, membership_map)`` where ``membership_map`` has shape
        ``(*data.shape, 3)`` in ascending-center order (CSF, GM, WM).
    """
    foreground = data[foreground_mask]
    if foreground.size > max_samples:
        rng = np.random.default_rng(seed)
        fit_data = rng.choice(foreground, size=max_samples, replace=False)
    else:
        fit_data = foreground
    centers, _ = _fcm.fuzzy_cmeans(fit_data, n_classes=3, seed=seed)
    memberships = _fcm.predict_memberships(foreground, centers)
    membership_map = np.zeros((*data.shape, 3), dtype=np.float32)
    for i in range(3):
        membership_map[..., i][foreground_mask] = memberships[i]
    means = np.array(
        [np.average(data[foreground_mask], weights=memberships[i]) for i in range(3)],
        dtype=np.float32,
    )
    return means, membership_map


def fcm_array(
    data: IntensityArray,
    foreground: BinaryMask,
    *,
    modality: str = "t1",
    tissue: str = "wm",
    membership: IntensityArray | None = None,
    norm_value: float = 1.0,
    seed: int | None = 0,
) -> IntensityArray:
    """Normalize an intensity array to the fuzzy c-means mean of a tissue class.

    For T1-w images, three-class fuzzy c-means segments the foreground into
    CSF/GM/WM memberships and the array is scaled so the ``tissue`` mean
    equals ``norm_value``. This is the recommended starting point for T1-w
    brain images.

    Args:
        data: intensity array.
        foreground: boolean foreground (brain) mask.
        modality: "t1" computes memberships from ``data`` itself. For other
            modalities, pass ``membership`` (from a co-registered T1-w image,
            e.g. via :func:`intensity_normalization.tissue_membership`);
            otherwise ``foreground`` is used as hard tissue weights.
        tissue: "csf", "gm", or "wm".
        membership: precomputed tissue membership map (same shape as data).
        norm_value: intensity the tissue mean is mapped to.
        seed: RNG seed for the FCM fit; ``None`` is nondeterministic.

    Returns:
        The normalized intensity array.
    """
    weights: IntensityArray
    if membership is not None:
        if membership.shape != data.shape:
            msg = (
                f"Membership shape {membership.shape} does not match image shape "
                f"{data.shape}. It must come from a co-registered image."
            )
            raise IntensityNormalizationError(msg)
        weights = np.asarray(membership, dtype=np.float32)
    elif modality.lower() == "t1":
        _, membership_map = tissue_means(data, foreground, seed=seed)
        weights = membership_map[..., _tissue_index(tissue)]
    else:
        weights = foreground.astype(np.float32)

    tissue_mean = float(np.average(data[foreground], weights=weights[foreground]))
    if tissue_mean == 0.0:
        msg = f"The {tissue} mean is zero; cannot scale by it. Check the image and mask."
        raise IntensityNormalizationError(msg)
    return standardize(data, 0.0, tissue_mean, norm_value)


def fcm(
    image: Image,
    mask: Mask | None = None,
    *,
    modality: str = "t1",
    tissue: str = "wm",
    membership: IntensityArray | None = None,
    norm_value: float = 1.0,
    seed: int | None = 0,
) -> Image:
    """Normalize an MR image (numpy or nibabel); see :func:`fcm_array`.

    Args:
        image: numpy array or nibabel image; the same type is returned.
        mask: foreground (brain) mask. If None, estimated as positive voxels.
        modality: "t1" computes memberships from ``image`` itself. For other
            modalities, pass ``membership`` (from a co-registered T1-w image,
            e.g. via :func:`intensity_normalization.tissue_membership`) or a
            ``mask`` to use as hard tissue weights.
        tissue: "csf", "gm", or "wm".
        membership: precomputed tissue membership map (same shape as image).
        norm_value: intensity the tissue mean is mapped to.
        seed: RNG seed for the FCM fit; ``None`` is nondeterministic.

    Returns:
        The normalized image, same type as ``image``.
    """
    data, restore = _image.unwrap(image)
    if modality.lower() != "t1" and membership is None and mask is None:
        msg = (
            f"FCM tissue memberships are only meaningful on T1-w images; got "
            f"modality={modality!r}. Pass membership= from a co-registered T1-w "
            "image (see intensity_normalization.tissue_membership) or a mask "
            "to use as tissue weights."
        )
        raise IntensityNormalizationError(msg)
    foreground = _image.resolve_foreground(data, _image.unwrap_mask(image, mask))
    normalized = fcm_array(
        data,
        foreground,
        modality=modality,
        tissue=tissue,
        membership=membership,
        norm_value=norm_value,
        seed=seed,
    )
    return restore(normalized)
