"""Tissue membership maps of a T1-w brain image via fuzzy c-means."""

from __future__ import annotations

import numpy as np

from intensity_normalization import _image
from intensity_normalization._image import ImageLike
from intensity_normalization.methods.fcm import tissue_means

__all__ = ["tissue_membership"]


def tissue_membership(
    image: ImageLike,
    /,
    mask: ImageLike | None = None,
    *,
    hard_segmentation: bool = False,
    seed: int | None = 0,
) -> ImageLike:
    """Fuzzy c-means tissue memberships of a T1-w brain image.

    Args:
        image: T1-w numpy array or nibabel image; the same type is returned.
        mask: foreground (brain) mask. If None, estimated as positive voxels.
        hard_segmentation: return a hard 3D label map (0 background, 1 CSF,
            2 GM, 3 WM) instead of per-class membership maps.
        seed: RNG seed for the FCM fit; ``None`` is nondeterministic.

    Returns:
        A 4D membership map, ``image.shape + (3,)`` in CSF/GM/WM order
        (see :data:`intensity_normalization.methods.fcm.TISSUES`), or a 3D
        label map with ``hard_segmentation=True``.
    """
    data, restore = _image.unwrap(image)
    mask_data = _image.unwrap(mask)[0] if mask is not None else None
    foreground_mask = _image.get_mask(data, mask_data)
    _, membership_map = tissue_means(data, foreground_mask, seed=seed)
    if hard_segmentation:
        labels = np.zeros(data.shape, dtype=np.float32)
        labels[foreground_mask] = membership_map[foreground_mask].argmax(axis=1) + 1
        return restore(labels)
    return restore(membership_map)
