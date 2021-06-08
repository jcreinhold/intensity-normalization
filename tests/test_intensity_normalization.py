#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `intensity_normalization` package."""

from pathlib import Path
from typing import List

import numpy as np
import nibabel as nib
import pytest

from intensity_normalization.cli import (
    fcm_main,
    histogram_main,
    kde_main,
    lsq_main,
    nyul_main,
    preprocessor_main,
    ravel_main,
    register_main,
    tissue_main,
    ws_main,
    zs_main,
)


@pytest.fixture
def cwd() -> Path:
    return Path.cwd().resolve()


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory) -> Path:
    return Path(tmpdir_factory.mktemp("out"))


@pytest.fixture(scope="session")
def image(temp_dir: Path) -> Path:
    image_data = np.random.randn(5, 5, 5)
    image_path = temp_dir / "test_image.nii"
    image = nib.Nifti1Image(image_data, np.eye(4))
    image.to_filename(image_path)
    return image_path


@pytest.fixture(scope="session")
def mask(temp_dir: Path) -> Path:
    mask_data = np.random.randint(0, 2, (5, 5, 5))
    mask_path = temp_dir / "test_mask.nii"
    mask = nib.Nifti1Image(mask_data, np.eye(4))
    mask.to_filename(mask_path)
    return mask_path


@pytest.fixture
def base_cli_image_args(image: Path, mask: Path, temp_dir: Path):
    return f"{image} -m {mask}".split()


@pytest.fixture
def base_cli_dir_args(temp_dir: Path):
    return f"{temp_dir} -m {temp_dir}".split()


def test_zscore_normalization_cli(base_cli_image_args: List[str]):
    args = base_cli_image_args + []
    retval = zs_main(args)
    assert retval == 0
