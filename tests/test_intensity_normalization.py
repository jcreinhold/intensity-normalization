#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `intensity_normalization` package."""

import os
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import pytest

from intensity_normalization.cli.fcm import fcm_main
from intensity_normalization.cli.histogram import histogram_main as hist_main
from intensity_normalization.cli.kde import kde_main
from intensity_normalization.cli.lsq import lsq_main
from intensity_normalization.cli.nyul import nyul_main
from intensity_normalization.cli.tissue_membership import (
    tissue_membership_main as tm_main,
)
from intensity_normalization.cli.whitestripe import whitestripe_main as ws_main
from intensity_normalization.cli.zscore import zscore_main as zs_main

try:
    import ants
except (ImportError, ModuleNotFoundError):
    ants = None
else:
    from intensity_normalization.cli.coregister import coregister_main
    from intensity_normalization.cli.preprocess import preprocess_main
    from intensity_normalization.cli.ravel import ravel_main

ANTSPY_DIR = Path.home() / ".antspy"
ANTSPY_DIR_EXISTS = ANTSPY_DIR.is_dir()


@pytest.fixture
def cwd() -> Path:
    return Path.cwd().resolve()


@pytest.fixture
def temp_dir(tmpdir_factory) -> Path:  # type: ignore[no-untyped-def]
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


@pytest.fixture
def out_dir(temp_dir: Path) -> Path:
    mask_dir = temp_dir / "normalized"
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
def base_cli_image_args(image: Path, mask: Path) -> List[str]:
    return f"{image} -m {mask}".split()


@pytest.fixture
def base_cli_dir_args(image: Path, mask: Path) -> List[str]:
    # use image, mask instead of image_dir, mask_dir so they are created
    return f"{image.parent} -m {mask.parent}".split()


def test_fcm_normalization_cli(base_cli_image_args: List[str]) -> None:
    args = base_cli_image_args
    retval = fcm_main(args)
    assert retval == 0


def test_fcm_normalization_nont1w_cli(image: Path, mask: Path) -> None:
    args = f"{image} -tm {mask} -mo t2".split()
    retval = fcm_main(args)
    assert retval == 0


def test_kde_normalization_cli(base_cli_image_args: List[str]) -> None:
    retval = kde_main(base_cli_image_args)
    assert retval == 0


def test_ws_normalization_cli(base_cli_image_args: List[str]) -> None:
    retval = ws_main(base_cli_image_args)
    assert retval == 0


def test_zscore_normalization_cli(base_cli_image_args: List[str]) -> None:
    retval = zs_main(base_cli_image_args)
    assert retval == 0


def test_lsq_normalization_cli(image: Path, mask: Path, out_dir: Path) -> None:
    args = f"{image.parent} -m {mask.parent} -o {out_dir}".split()
    retval = lsq_main(args)
    assert retval == 0
    os.remove(out_dir / "test_image_lsq.nii")
    args = f"{image.parent} -tm {out_dir} -mo t2".split()
    retval = lsq_main(args)
    assert retval == 0


def test_lsq_normalization_save_load_cli(
    image: Path,
    mask: Path,
    out_dir: Path,
    temp_dir: Path,
) -> None:
    base_args = f"{image.parent} -m {mask.parent} -o {out_dir}".split()
    args = base_args + ["-sstm", f"{temp_dir}/test.npy"]
    retval = lsq_main(args)
    assert retval == 0
    args = base_args + ["-lstm", f"{temp_dir}/test.npy"]
    retval = lsq_main(args)
    assert retval == 0


def test_nyul_normalization_cli(base_cli_dir_args: List[str]) -> None:
    retval = nyul_main(base_cli_dir_args)
    assert retval == 0


def test_nyul_normalization_save_load_cli(
    image: Path,
    mask: Path,
    out_dir: Path,
    temp_dir: Path,
) -> None:
    base_args = f"{image.parent} -m {mask.parent} -o {out_dir}".split()
    args = base_args + ["-ssh", f"{temp_dir}/test.npy"]
    retval = nyul_main(args)
    assert retval == 0
    args = base_args + ["-lsh", f"{temp_dir}/test.npy"]
    retval = nyul_main(args)
    assert retval == 0


@pytest.mark.skipif(ants is None, reason="Requires ANTsPy")
def test_ravel_normalization_cli(base_cli_dir_args: List[str]) -> None:
    retval = ravel_main(base_cli_dir_args)
    assert retval == 0


@pytest.fixture
def coregister_cli_args(image: Path) -> List[str]:
    return f"{image}".split()


@pytest.mark.skipif(ants is None, reason="Requires ANTsPy")
@pytest.mark.skipif(not ANTSPY_DIR_EXISTS, reason="ANTsPy directory wasn't found.")
def test_coregister_mni_cli(coregister_cli_args: List[str]) -> None:
    retval = coregister_main(coregister_cli_args)
    assert retval == 0


@pytest.fixture
def coregister_template_cli_args(image: Path, mask: Path) -> List[str]:
    return f"{image} -t {mask}".split()


@pytest.mark.skip("Test images are problematic.")
def test_coregister_template_cli(coregister_template_cli_args: List[str]) -> None:
    retval = coregister_main(coregister_template_cli_args)
    assert retval == 0


@pytest.fixture
def histogram_cli_args(base_cli_dir_args: List[str], temp_dir: Path) -> List[str]:
    return base_cli_dir_args + f"-o {temp_dir}/hist.png".split()


def test_histogram_cli(histogram_cli_args: List[str]) -> None:
    retval = hist_main(histogram_cli_args)
    assert retval == 0


@pytest.fixture
def preprocess_cli_args(image: Path) -> List[str]:
    return f"{image} -2n4".split()


@pytest.mark.skip("Takes too long.")
def test_preprocess_cli(preprocess_cli_args: List[str]) -> None:
    retval = preprocess_main(preprocess_cli_args)
    assert retval == 0


@pytest.fixture
def tissue_membership_cli_args(image: Path) -> List[str]:
    return f"{image}".split()


def test_tissue_membership_cli(tissue_membership_cli_args: List[str]) -> None:
    retval = tm_main(tissue_membership_cli_args)
    assert retval == 0


def test_tissue_membership_hard_seg_cli(tissue_membership_cli_args: List[str]) -> None:
    tissue_membership_cli_args.append("-hs")
    retval = tm_main(tissue_membership_cli_args)
    assert retval == 0
