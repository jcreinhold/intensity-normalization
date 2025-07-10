"""I/O adapters for file operations."""

import os
from pathlib import Path

from intensity_normalization.adapters.images import SUPPORTED_EXTENSIONS, create_image
from intensity_normalization.domain.exceptions import ImageLoadError
from intensity_normalization.domain.protocols import ImageProtocol


def load_image(path: str | os.PathLike) -> ImageProtocol:
    """Load image from file path."""
    return create_image(path)


def save_image(image: ImageProtocol, path: str | os.PathLike) -> None:
    """Save image to file path."""
    image.save(path)


def is_supported_format(path: str | os.PathLike) -> bool:
    """Check if file extension is supported."""
    path_obj = Path(path)

    # Handle .nii.gz specifically
    if str(path_obj).endswith(".nii.gz"):
        return True

    return path_obj.suffix in SUPPORTED_EXTENSIONS


def validate_input_path(path: str | os.PathLike) -> Path:
    """Validate input file path exists and is supported format."""
    path_obj = Path(path)

    if not path_obj.exists():
        raise ImageLoadError(f"Input file {path_obj} does not exist")

    if not is_supported_format(path_obj):
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ImageLoadError(f"Unsupported file format. Supported: {supported_list}")

    return path_obj


def generate_output_path(input_path: str | os.PathLike, method: str) -> Path:
    """Generate output filename based on input and method."""
    input_path = Path(input_path)
    stem = input_path.stem

    # Handle .nii.gz case
    if stem.endswith(".nii"):
        stem = stem[:-4]

    suffix = input_path.suffix
    if input_path.name.endswith(".nii.gz"):
        suffix = ".nii.gz"

    return input_path.parent / f"{stem}_{method}{suffix}"
