"""Pytest configuration and fixtures for intensity-normalization tests."""

import tempfile
from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from intensity_normalization.adapters.images import NibabelImageAdapter, NumpyImageAdapter
from intensity_normalization.domain.protocols import ImageProtocol


@pytest.fixture
def sample_3d_data() -> npt.NDArray[np.floating]:
    """Generate synthetic 3D brain-like data."""
    import numpy as np

    shape = (64, 64, 32)
    data = np.zeros(shape, dtype=np.float32)

    # Create realistic MR intensities with tissue-like clusters
    #   Background -> ~0
    #   CSF -> low intensity ~200
    data[10:20, 10:20, 10:20] = np.random.normal(200, 20, (10, 10, 10))
    # GM (medium intensity ~600)
    data[20:40, 20:40, 10:20] = np.random.normal(600, 30, (20, 20, 10))
    # WM (high intensity ~1000)
    data[40:60, 40:60, 10:20] = np.random.normal(1000, 40, (20, 20, 10))

    # Ensure non-negative values
    return np.clip(data, 0, None)


@pytest.fixture
def sample_mask() -> npt.NDArray[np.floating]:
    """Generate brain mask."""
    import numpy as np

    shape = (64, 64, 32)
    mask = np.zeros(shape, dtype=np.float32)
    mask[10:60, 10:60, 10:20] = 1.0
    return mask


@pytest.fixture
def numpy_image(sample_3d_data: npt.NDArray[np.floating]) -> NumpyImageAdapter:
    """Create numpy image adapter."""
    return NumpyImageAdapter(sample_3d_data)


@pytest.fixture
def numpy_mask(sample_mask: npt.NDArray[np.floating]) -> NumpyImageAdapter:
    """Create numpy mask adapter."""
    return NumpyImageAdapter(sample_mask)


@pytest.fixture
def nibabel_image(sample_3d_data: npt.NDArray[np.floating]) -> NibabelImageAdapter:
    """Create nibabel image adapter."""
    import nibabel as nib
    import numpy as np

    affine = np.eye(4)
    nib_img = nib.nifti1.Nifti1Image(sample_3d_data, affine)
    return NibabelImageAdapter(nib_img)


@pytest.fixture
def nibabel_mask(sample_mask: npt.NDArray[np.floating]) -> NibabelImageAdapter:
    """Create nibabel mask adapter."""
    import nibabel as nib
    import numpy as np

    affine = np.eye(4)
    nib_img = nib.nifti1.Nifti1Image(sample_mask, affine)
    return NibabelImageAdapter(nib_img)


@pytest.fixture
def temp_nifti_file(sample_3d_data: npt.NDArray[np.floating]) -> Generator[Path, None, None]:
    """Create temporary NIfTI file."""
    import nibabel as nib
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        affine = np.eye(4)
        img = nib.nifti1.Nifti1Image(sample_3d_data, affine)
        nib.loadsave.save(img, f.name)
        yield Path(f.name)
        Path(f.name).unlink()


@pytest.fixture
def temp_mask_file(sample_mask: npt.NDArray[np.floating]) -> Generator[Path, None, None]:
    """Create temporary mask NIfTI file."""
    import nibabel as nib
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        affine = np.eye(4)
        img = nib.nifti1.Nifti1Image(sample_mask, affine)
        nib.loadsave.save(img, f.name)
        yield Path(f.name)
        Path(f.name).unlink()


@pytest.fixture(params=["numpy", "nibabel"])
def image_fixture(
    request: pytest.FixtureRequest,
    numpy_image: NumpyImageAdapter,
    nibabel_image: NibabelImageAdapter,
) -> ImageProtocol:
    """Parametrized fixture that returns either numpy or nibabel image."""
    if request.param == "numpy":
        return numpy_image
    return nibabel_image


@pytest.fixture(params=["numpy", "nibabel"])
def mask_fixture(
    request: pytest.FixtureRequest,
    numpy_mask: NumpyImageAdapter,
    nibabel_mask: NibabelImageAdapter,
) -> ImageProtocol:
    """Parametrized fixture that returns either numpy or nibabel mask."""
    if request.param == "numpy":
        return numpy_mask
    return nibabel_mask


@pytest.fixture
def multiple_images(sample_3d_data: npt.NDArray[np.floating]) -> Sequence[ImageProtocol]:
    """Create multiple images for population-based testing."""
    import numpy as np

    images = []
    for _ in range(3):
        # Add some variation to each image
        noise = np.random.normal(0, 50, sample_3d_data.shape)
        data = sample_3d_data + noise
        data = np.clip(data, 0, None)  # Ensure non-negative
        images.append(NumpyImageAdapter(data))

    return images


@pytest.fixture
def multiple_masks(sample_mask: npt.NDArray[np.floating]) -> list[ImageProtocol]:
    """Create multiple masks for population-based testing."""
    return [NumpyImageAdapter(sample_mask) for _ in range(3)]


# Legacy fixtures for backwards compatibility
@pytest.fixture
def cwd() -> Path:
    return Path.cwd().resolve()


@pytest.fixture
def temp_dir(tmpdir_factory: pytest.TempdirFactory) -> Path:
    return Path(tmpdir_factory.mktemp("out"))


@pytest.fixture
def image_dir(temp_dir: Path) -> Path:
    image_dir = temp_dir / "image"
    image_dir.mkdir()
    return image_dir


@pytest.fixture
def image(image_dir: Path) -> Path:
    import nibabel as nib
    import numpy as np

    image_data = np.random.randn(5, 5, 5)
    image_path = image_dir / "test_image.nii"
    image = nib.nifti1.Nifti1Image(image_data, np.eye(4))
    image.to_filename(image_path)
    return image_path


@pytest.fixture
def mask_dir(temp_dir: Path) -> Path:
    mask_dir = temp_dir / "mask"
    mask_dir.mkdir()
    return mask_dir


@pytest.fixture
def out_dir(temp_dir: Path) -> Path:
    out_dir = temp_dir / "normalized"
    out_dir.mkdir()
    return out_dir


@pytest.fixture
def mask(mask_dir: Path) -> Path:
    import nibabel as nib
    import numpy as np

    mask_data = np.random.randint(0, 2, (5, 5, 5)).astype(np.float32)
    mask_path = mask_dir / "test_mask.nii"
    mask = nib.nifti1.Nifti1Image(mask_data, np.eye(4))
    mask.to_filename(mask_path)
    return mask_path


@pytest.fixture
def base_cli_image_args(image: Path, mask: Path) -> list[str]:
    return f"{image} -m {mask}".split()


@pytest.fixture
def base_cli_dir_args(image: Path, mask: Path) -> list[str]:
    # use image, mask instead of image_dir, mask_dir so they are created
    return f"{image.parent} -m {mask.parent}".split()
