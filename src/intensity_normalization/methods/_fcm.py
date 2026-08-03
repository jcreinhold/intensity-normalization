"""Fuzzy c-means clustering on 1D intensity data (private).

In-house replacement for ``scikit-fuzzy`` (unmaintained; broke on Python 3.12).
Only the small part of FCM this package needs is implemented: 1D data, a fixed
number of clusters, and probabilistic memberships. Clusters are always returned
sorted by ascending center intensity, so for a T1-w brain image the classes map
to CSF / GM / WM in that order.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["fuzzy_cmeans", "predict_memberships"]


def _sort_by_center(
    centers: npt.NDArray[np.floating], memberships: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    order = np.argsort(centers)
    return centers[order], memberships[order]


def predict_memberships(
    data: npt.NDArray[np.floating],
    centers: npt.NDArray[np.floating],
    *,
    m: float = 2.0,
) -> npt.NDArray[np.floating]:
    """Compute fuzzy memberships of ``data`` to fixed ``centers``.

    Args:
        data: 1D array of intensities.
        centers: cluster centers, shape ``(n_classes,)``.
        m: fuzziness parameter (> 1); 2 is standard.

    Returns:
        Memberships of shape ``(n_classes, data.size)``; columns sum to 1.
    """
    x = np.asarray(data, dtype=np.float32).ravel()
    c = np.asarray(centers, dtype=np.float32).ravel()
    dist = np.abs(x[None, :] - c[:, None])
    # A sample exactly at a center must get membership 1 there; clipping makes
    # its inverse distance dominate instead of producing NaNs.
    np.maximum(dist, np.finfo(np.float32).eps, out=dist)
    inv = dist ** (-2.0 / (m - 1.0))
    return inv / inv.sum(axis=0, keepdims=True)


def fuzzy_cmeans(
    data: npt.NDArray[np.floating],
    n_classes: int = 3,
    *,
    m: float = 2.0,
    tolerance: float = 5e-3,
    max_iterations: int = 50,
    seed: int | None = 0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Cluster 1D intensities with fuzzy c-means.

    Args:
        data: 1D array of intensities (e.g., foreground voxels).
        n_classes: number of clusters (3 for CSF/GM/WM).
        m: fuzziness parameter (> 1); 2 is standard.
        tolerance: stop when memberships change by less than this (max norm).
        max_iterations: iteration cap.
        seed: RNG seed for membership initialization; ``None`` is nondeterministic.

    Returns:
        ``(centers, memberships)`` with centers sorted ascending and
        memberships of shape ``(n_classes, data.size)`` in the same order.

    Raises:
        ValueError: if ``n_classes`` < 1, ``m`` <= 1, or data has fewer than
            ``n_classes`` distinct values.
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be positive. Got {n_classes}.")
    if m <= 1.0:
        raise ValueError(f"m must be greater than 1. Got {m}.")
    x = np.asarray(data, dtype=np.float32).ravel()
    if np.unique(x).size < n_classes:
        raise ValueError(
            f"Cannot find {n_classes} classes in data with "
            f"{np.unique(x).size} distinct values. Is the foreground non-empty?"
        )

    rng = np.random.default_rng(seed)
    u = rng.random((n_classes, x.size), dtype=np.float32)
    u /= u.sum(axis=0, keepdims=True)

    centers = np.zeros(n_classes, dtype=np.float32)
    for _ in range(max_iterations):
        um = u**m
        centers = (um @ x) / um.sum(axis=1)
        u_new = predict_memberships(x, centers, m=m)
        if np.abs(u_new - u).max() < tolerance:
            u = u_new
            break
        u = u_new

    return _sort_by_center(centers, u)
