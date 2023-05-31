"""Tests for ants-required `intensity_normalization` functions."""

import pathlib
import typing

import numpy as np
import pytest

try:
    from intensity_normalization.cli.coregister import coregister_main
    from intensity_normalization.cli.preprocess import preprocess_main
    from intensity_normalization.cli.ravel import ravel_main
except RuntimeError:
    pytest.skip("ANTsPy required for these tests. Skipping.", allow_module_level=True)


ANTSPY_DIR = pathlib.Path.home() / ".antspy"
ANTSPY_DIR_EXISTS = ANTSPY_DIR.is_dir()


@pytest.fixture
def coregister_cli_args(image: pathlib.Path) -> typing.List[str]:
    return f"{image}".split()


@pytest.mark.skipif(not ANTSPY_DIR_EXISTS, reason="ANTsPy directory wasn't found.")
def test_coregister_mni_cli(coregister_cli_args: typing.List[str]) -> None:
    retval = coregister_main(coregister_cli_args)
    assert retval == 0


@pytest.fixture
def coregister_template_cli_args(
    image: pathlib.Path, mask: pathlib.Path
) -> typing.List[str]:
    return f"{image} -t {mask}".split()


@pytest.mark.skip("Test images are problematic.")
def test_coregister_template_cli(
    coregister_template_cli_args: typing.List[str],
) -> None:
    retval = coregister_main(coregister_template_cli_args)
    assert retval == 0


@pytest.fixture
def preprocess_cli_args(image: pathlib.Path) -> typing.List[str]:
    return f"{image} -2n4".split()


@pytest.mark.skip("Takes too long.")
def test_preprocess_cli(preprocess_cli_args: typing.List[str]) -> None:
    retval = preprocess_main(preprocess_cli_args)
    assert retval == 0


def test_ravel_normalization_cli(base_cli_dir_args: typing.List[str]) -> None:
    args = base_cli_dir_args + ["--membership-threshold", "0.1"]
    np.random.seed(1337)
    retval = ravel_main(args)
    assert retval == 0
