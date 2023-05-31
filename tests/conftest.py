import pathlib
import typing

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def cwd() -> pathlib.Path:
    return pathlib.Path.cwd().resolve()


@pytest.fixture
def temp_dir(tmpdir_factory: pytest.TempdirFactory) -> pathlib.Path:
    return pathlib.Path(tmpdir_factory.mktemp("out"))


@pytest.fixture
def image_dir(temp_dir: pathlib.Path) -> pathlib.Path:
    image_dir = temp_dir / "image"
    image_dir.mkdir()
    return image_dir


@pytest.fixture
def image(image_dir: pathlib.Path) -> pathlib.Path:
    image_data = np.random.randn(5, 5, 5)
    image_path = image_dir / "test_image.nii"
    image = nib.nifti1.Nifti1Image(image_data, np.eye(4))
    image.to_filename(image_path)
    return image_path


@pytest.fixture
def mask_dir(temp_dir: pathlib.Path) -> pathlib.Path:
    mask_dir = temp_dir / "mask"
    mask_dir.mkdir()
    return mask_dir


@pytest.fixture
def out_dir(temp_dir: pathlib.Path) -> pathlib.Path:
    mask_dir = temp_dir / "normalized"
    mask_dir.mkdir()
    return mask_dir


@pytest.fixture
def mask(mask_dir: pathlib.Path) -> pathlib.Path:
    mask_data: np.ndarray = np.random.randint(0, 2, (5, 5, 5)).astype(np.float32)
    mask_path = mask_dir / "test_mask.nii"
    mask = nib.nifti1.Nifti1Image(mask_data, np.eye(4))
    mask.to_filename(mask_path)
    return mask_path


@pytest.fixture
def base_cli_image_args(image: pathlib.Path, mask: pathlib.Path) -> typing.List[str]:
    return f"{image} -m {mask}".split()


@pytest.fixture
def base_cli_dir_args(image: pathlib.Path, mask: pathlib.Path) -> typing.List[str]:
    # use image, mask instead of image_dir, mask_dir so they are created
    return f"{image.parent} -m {mask.parent}".split()
