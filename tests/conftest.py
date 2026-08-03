"""Shared fixtures: synthetic brain phantoms with known tissue statistics."""

from __future__ import annotations

import pathlib

import nibabel as nib
import numpy as np
import pytest

TISSUE_MEANS = {1: 100.0, 2: 300.0, 3: 500.0}  # csf, gm, wm


def make_phantom(
    shape: tuple[int, int, int] = (30, 30, 30),
    *,
    scale: float = 1.0,
    shift: float = 0.0,
    noise: float = 10.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T1-w-like phantom: background 0, csf/gm/wm at known means (x scale + shift).

    Returns (image, foreground_mask, labels).
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.uint8)
    a, b = shape[0] // 4, shape[0] // 8
    c = int(shape[0] * 0.4)
    labels[a:-a, a:-a, a:-a] = 1
    inner = labels[b:-b, b:-b, b:-b]
    inner[inner == 0] = 2
    labels[c:-c, c:-c, c:-c] = 3
    image = np.zeros(shape, np.float32)
    for k, mean in TISSUE_MEANS.items():
        n = int((labels == k).sum())
        image[labels == k] = rng.normal(mean * scale + shift, noise, n)
    return image, (labels > 0), labels


def make_population(
    n: int = 4,
    shape: tuple[int, int, int] = (24, 24, 24),
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Same phantom at different global scales/shifts, with masks."""
    params = [(1.0, 0.0), (1.3, 40.0), (0.7, -20.0), (2.0, 100.0), (1.1, 10.0)][:n]
    images, masks = [], []
    for i, (scale, shift) in enumerate(params):
        image, mask, _ = make_phantom(shape, scale=scale, shift=shift, seed=i)
        images.append(image)
        masks.append(mask.astype(np.float32))
    return images, masks


@pytest.fixture(scope="session")
def phantom() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_phantom()


@pytest.fixture(scope="session")
def population() -> tuple[list[np.ndarray], list[np.ndarray]]:
    return make_population()


@pytest.fixture(scope="session")
def nifti_dir(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, pathlib.Path]:
    """A directory of NIfTI images + masks from the population fixture."""
    images, masks = make_population(3)
    image_dir = tmp_path_factory.mktemp("images")
    mask_dir = tmp_path_factory.mktemp("masks")
    for i, (image, mask) in enumerate(zip(images, masks, strict=True)):
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_dir / f"sub{i}.nii.gz")
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_dir / f"sub{i}.nii.gz")
    return image_dir, mask_dir
