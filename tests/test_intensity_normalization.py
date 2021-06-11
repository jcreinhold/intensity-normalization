#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `intensity_normalization` package."""

from pathlib import Path
from typing import List

import numpy as np
import nibabel as nib
import pytest

try:
    import ants
except (ImportError, ModuleNotFoundError):
    ants = None

ANTSPY_DIR = Path.home() / ".antspy"
ANTSPY_DIR_EXISTS = ANTSPY_DIR.is_dir()

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


@pytest.fixture
def temp_dir(tmpdir_factory) -> Path:
    return Path(tmpdir_factory.mktemp("out"))


@pytest.fixture
def image_dir(temp_dir: Path) -> Path:
    image_dir = temp_dir / "image"
    image_dir.mkdir()
    return image_dir


@pytest.fixture()
def image(image_dir: Path) -> Path:
    image_data = np.random.randn(5, 5, 5)
    image_path = image_dir / "test_image.nii"
    image = nib.Nifti1Image(image_data, np.eye(4))
    image.to_filename(image_path)
    return image_path


@pytest.fixture
def mask_dir(temp_dir: Path) -> Path:
    mask_dir = temp_dir / "mask"
    mask_dir.mkdir()
    return mask_dir


@pytest.fixture()
def mask(mask_dir: Path) -> Path:
    mask_data = np.random.randint(0, 2, (5, 5, 5)).astype(float)
    mask_path = mask_dir / "test_mask.nii"
    mask = nib.Nifti1Image(mask_data, np.eye(4))
    mask.to_filename(mask_path)
    return mask_path


@pytest.fixture
def base_cli_image_args(image: Path, mask: Path):
    return f"{image} -m {mask}".split()


@pytest.fixture
def base_cli_dir_args(image: Path, mask: Path):
    return f"{image.parent} -m {mask.parent}".split()


def test_fcm_normalization_cli(base_cli_image_args: List[str]):
    args = base_cli_image_args
    retval = fcm_main(args)
    assert retval == 0


def test_kde_normalization_cli(base_cli_image_args: List[str]):
    retval = kde_main(base_cli_image_args)
    assert retval == 0


def test_ws_normalization_cli(base_cli_image_args: List[str]):
    retval = ws_main(base_cli_image_args)
    assert retval == 0


def test_zscore_normalization_cli(base_cli_image_args: List[str]):
    retval = zs_main(base_cli_image_args)
    assert retval == 0


def test_lsq_normalization_cli(base_cli_dir_args: List[str]):
    retval = lsq_main(base_cli_dir_args)
    assert retval == 0


def test_nyul_normalization_cli(base_cli_dir_args: List[str]):
    retval = nyul_main(base_cli_dir_args)
    assert retval == 0


@pytest.mark.skip("Not implemented.")
def test_ravel_normalization_cli(base_cli_dir_args: List[str]):
    retval = ravel_main(base_cli_dir_args)
    assert retval == 0


@pytest.fixture
def coregister_cli_args(image: Path):
    return f"{image}".split()


@pytest.mark.skipif(ants is None, reason="Requires ANTsPy")
@pytest.mark.skipif(not ANTSPY_DIR_EXISTS, reason="ANTsPy directory wasn't found.")
def test_coregister_mni_cli(coregister_cli_args: List[str]):
    retval = register_main(coregister_cli_args)
    assert retval == 0


@pytest.fixture
def coregister_template_cli_args(image: Path, mask: Path):
    return f"{image} -t {mask}".split()


@pytest.mark.skip("Test images are problematic.")
def test_coregister_template_cli(coregister_template_cli_args: List[str]):
    retval = register_main(coregister_template_cli_args)
    assert retval == 0


@pytest.fixture
def histogram_cli_args(base_cli_dir_args: List[str], temp_dir: Path):
    return base_cli_dir_args + f"-o {temp_dir}/hist.png".split()


def test_histogram_cli(histogram_cli_args: List[str]):
    retval = histogram_main(histogram_cli_args)
    assert retval == 0


@pytest.fixture
def preprocess_cli_args(image: Path):
    return f"{image} -2n4".split()


@pytest.mark.skip("Test images do not work.")
def test_preprocess_cli(preprocess_cli_args: List[str]):
    retval = preprocessor_main(preprocess_cli_args)
    assert retval == 0


@pytest.fixture
def tissue_membership_cli_args(image: Path):
    return f"{image}".split()


def test_tissue_membership_cli(tissue_membership_cli_args: List[str]):
    retval = tissue_main(tissue_membership_cli_args)
    assert retval == 0


def test_tissue_membership_hard_seg_cli(tissue_membership_cli_args: List[str]):
    tissue_membership_cli_args.append("-hs")
    retval = tissue_main(tissue_membership_cli_args)
    assert retval == 0
