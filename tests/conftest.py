"""Shared fixtures built on the phantom helpers in tests/phantoms.py."""

from __future__ import annotations

import pathlib

import nibabel as nib
import numpy as np
import pytest
from phantoms import make_phantom, make_population


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
