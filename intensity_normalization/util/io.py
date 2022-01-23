"""Input/output utilities for the project
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = [
    "gather_images",
    "gather_images_and_masks",
    "glob_ext",
    "split_filename",
]

import pathlib
import typing

import nibabel as nib

from intensity_normalization.typing import Array, NiftiImage, PathLike


def gather_images(
    dirpath: PathLike,
    ext: str = "nii*",
    return_data: bool = False,
) -> Union[List[NiftiImage], List[Array]]:
    """return all images of extension `ext` from a directory"""
    if isinstance(dirpath, str):
        dirpath = Path(dirpath)
    assert dirpath.is_dir()
    image_filenames = glob_ext(dirpath, ext)
    images = []
    for fn in image_filenames:
        image = nib.load(fn)
        if return_data:
            image = image.get_fdata()
        images.append(image)
    return images


def gather_images_and_masks(
    image_dir: PathLike,
    mask_dir: Optional[PathLike] = None,
    ext: str = "nii*",
    return_data: bool = False,
) -> Tuple[Union[List[Any], List[Array]], Union[List[Any], List[Any]]]:
    images = gather_images(image_dir, ext, return_data)
    if mask_dir is not None:
        masks = gather_images(mask_dir, ext, return_data)
    else:
        masks = [None] * len(images)
    return images, masks


def glob_ext(dirpath: PathLike, ext: str = "nii*") -> List[Path]:
    """return a sorted list of ext files for a given directory path"""
    if isinstance(dirpath, str):
        dirpath = Path(dirpath)
    assert dirpath.is_dir()
    filenames = sorted(dirpath.resolve().glob(f"*.{ext}"))
    return filenames


def split_filename(
    filepath: Union[str, Path],
    resolve: bool = False,
) -> Tuple[Path, str, str]:
    """split a filepath into the directory, base, and extension"""
    filepath = Path(filepath)
    if resolve:
        filepath = filepath.resolve()
    path = filepath.parent
    _base = Path(filepath.stem)
    ext = filepath.suffix
    if ext == ".gz":
        ext2 = _base.suffix
        base = str(_base.stem)
        ext = ext2 + ext
    else:
        base = str(_base)
    return Path(path), base, ext
