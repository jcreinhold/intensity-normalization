"""Path I/O for the CLI: load/save neuroimages and enumerate datasets.

The math API never sees paths; this module is the boundary that turns files
into nibabel images and back.
"""

from __future__ import annotations

import pathlib
import typing
from collections.abc import Collection, Sequence

import nibabel as nib
import nibabel.spatialimages  # explicit so nib.spatialimages resolves

from intensity_normalization.errors import IntensityNormalizationError

__all__ = [
    "IMAGE_EXTENSIONS",
    "find_images",
    "load_image",
    "match_masks",
    "output_path",
    "save_image",
    "split_filename",
]

#: extensions nibabel can read that we advertise in the CLI
IMAGE_EXTENSIONS: tuple[str, ...] = (".nii.gz", ".nii", ".mgz", ".mgh", ".mnc", ".img")


def load_image(path: str | pathlib.Path) -> nib.spatialimages.SpatialImage:
    """Load a neuroimage from disk (any nibabel-supported format)."""
    path = pathlib.Path(path)
    if not path.exists():
        raise IntensityNormalizationError(f"Image not found: {path}")
    try:
        return typing.cast(nib.spatialimages.SpatialImage, nib.loadsave.load(path))
    except Exception as exn:
        raise IntensityNormalizationError(f"Could not read {path} as a neuroimage: {exn}") from exn


def save_image(image: nib.spatialimages.SpatialImage, path: str | pathlib.Path) -> pathlib.Path:
    """Save a nibabel image to disk, creating parent directories."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.loadsave.save(image, path)
    return path


def split_filename(filepath: str | pathlib.Path) -> tuple[pathlib.Path, str, str]:
    """Split a path into ``(directory, base, extension)``; handles ``.nii.gz``.

    >>> split_filename("path/base.nii.gz")
    (PosixPath('path'), 'base', '.nii.gz')
    """
    filepath = pathlib.Path(filepath)
    if not str(filepath):
        raise ValueError("filepath must be non-empty.")
    base = filepath.stem
    ext = filepath.suffix
    if ext == ".gz":
        ext = pathlib.Path(base).suffix + ext
        base = pathlib.Path(base).stem
    return filepath.parent, base, ext


def find_images(
    directory: str | pathlib.Path,
    *,
    exclude: Collection[str] = (),
) -> list[pathlib.Path]:
    """Sorted neuroimage paths in a directory (non-recursive).

    Raises:
        IntensityNormalizationError: not a directory or no images found.
    """
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        raise IntensityNormalizationError(f"Not a directory: {directory}")
    paths = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file()
        and any(p.name.endswith(e) for e in IMAGE_EXTENSIONS)
        and all(exc not in p.name for exc in exclude)
    ]
    if not paths:
        exts = ", ".join(IMAGE_EXTENSIONS)
        raise IntensityNormalizationError(f"No neuroimages ({exts}) found in {directory}")
    return paths


def match_masks(
    image_paths: Sequence[pathlib.Path],
    mask_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Mask path for each image path, matched by filename within ``mask_dir``.

    Raises:
        IntensityNormalizationError: an image has no same-named mask, listing
            the first missing one.
    """
    mask_dir = pathlib.Path(mask_dir)
    if not mask_dir.is_dir():
        raise IntensityNormalizationError(f"Not a directory: {mask_dir}")
    available = {p.name for p in mask_dir.iterdir() if p.is_file()}
    masks = []
    for image_path in image_paths:
        if image_path.name not in available:
            raise IntensityNormalizationError(
                f"No mask named {image_path.name} in {mask_dir}. Masks must share filenames with their images."
            )
        masks.append(mask_dir / image_path.name)
    return masks


def output_path(
    image_path: str | pathlib.Path,
    *,
    suffix: str,
    output_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Output path for an input image: ``<base>_<suffix><ext>`` in the output dir."""
    directory, base, ext = split_filename(image_path)
    out_dir = pathlib.Path(output_dir) if output_dir is not None else directory
    return out_dir / f"{base}_{suffix}{ext}"
