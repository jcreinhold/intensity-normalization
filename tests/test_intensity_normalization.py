"""Tests for non-antspy `intensity_normalization` functions."""

import os
import pathlib
import typing

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


def test_fcm_normalization_cli(base_cli_image_args: typing.List[str]) -> None:
    retval = fcm_main(base_cli_image_args)
    assert retval == 0


def test_fcm_normalization_nont1w_cli(image: pathlib.Path, mask: pathlib.Path) -> None:
    args = f"{image} -tm {mask} -mo t2".split()
    retval = fcm_main(args)
    assert retval == 0


def test_kde_normalization_cli(base_cli_image_args: typing.List[str]) -> None:
    retval = kde_main(base_cli_image_args)
    assert retval == 0


def test_ws_normalization_cli(base_cli_image_args: typing.List[str]) -> None:
    retval = ws_main(base_cli_image_args)
    assert retval == 0


def test_zscore_normalization_cli(base_cli_image_args: typing.List[str]) -> None:
    retval = zs_main(base_cli_image_args)
    assert retval == 0


def test_lsq_normalization_cli(
    image: pathlib.Path, mask: pathlib.Path, out_dir: pathlib.Path
) -> None:
    args = f"{image.parent} -m {mask.parent} -o {out_dir}".split()
    retval = lsq_main(args)
    assert retval == 0
    os.remove(out_dir / "test_image_lsq.nii")
    args = f"{image.parent} -tm {out_dir} -mo t2".split()
    retval = lsq_main(args)
    assert retval == 0


def test_lsq_normalization_save_load_cli(
    image: pathlib.Path,
    mask: pathlib.Path,
    out_dir: pathlib.Path,
    temp_dir: pathlib.Path,
) -> None:
    base_args = f"{image.parent} -m {mask.parent} -o {out_dir}".split()
    args = base_args + ["-sstm", f"{temp_dir}/test.npy"]
    retval = lsq_main(args)
    assert retval == 0
    args = base_args + ["-lstm", f"{temp_dir}/test.npy"]
    retval = lsq_main(args)
    assert retval == 0


def test_nyul_normalization_cli(base_cli_dir_args: typing.List[str]) -> None:
    retval = nyul_main(base_cli_dir_args)
    assert retval == 0


def test_nyul_normalization_save_load_cli(
    image: pathlib.Path,
    mask: pathlib.Path,
    out_dir: pathlib.Path,
    temp_dir: pathlib.Path,
) -> None:
    base_args = f"{image.parent} -m {mask.parent} -o {out_dir}".split()
    args = base_args + ["-ssh", f"{temp_dir}/test.npy"]
    retval = nyul_main(args)
    assert retval == 0
    args = base_args + ["-lsh", f"{temp_dir}/test.npy"]
    retval = nyul_main(args)
    assert retval == 0


@pytest.fixture
def histogram_cli_args(
    base_cli_dir_args: typing.List[str], temp_dir: pathlib.Path
) -> typing.List[str]:
    return base_cli_dir_args + f"-o {temp_dir}/hist.png".split()


def test_histogram_cli(histogram_cli_args: typing.List[str]) -> None:
    retval = hist_main(histogram_cli_args)
    assert retval == 0


@pytest.fixture
def tissue_membership_cli_args(image: pathlib.Path) -> typing.List[str]:
    return f"{image}".split()


def test_tissue_membership_cli(tissue_membership_cli_args: typing.List[str]) -> None:
    retval = tm_main(tissue_membership_cli_args)
    assert retval == 0


def test_tissue_membership_hard_seg_cli(
    tissue_membership_cli_args: typing.List[str],
) -> None:
    tissue_membership_cli_args.append("-hs")
    retval = tm_main(tissue_membership_cli_args)
    assert retval == 0
