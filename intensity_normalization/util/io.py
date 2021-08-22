# -*- coding: utf-8 -*-
"""
intensity_normalization.util.io

assortment of input/output utilities for the project

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 1, 2021
"""

__all__ = [
    "gather_images",
    "gather_images_and_masks",
    "glob_ext",
    "split_filename",
]

from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib

from intensity_normalization.type import Array, NiftiImage, PathLike


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
) -> Union[
    Tuple[List[NiftiImage], List[Optional[NiftiImage]]],
    Tuple[List[Array], List[Optional[Array]]],
]:
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
