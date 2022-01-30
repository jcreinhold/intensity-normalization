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

import builtins
import pathlib
import typing

import pymedio.image as mioi

import intensity_normalization.typing as intnormt


def gather_images(
    dirpath: intnormt.PathLike,
    *,
    ext: builtins.str = "nii*",
) -> typing.List[mioi.Image]:
    """return all images of extension `ext` from a directory"""
    if not isinstance(dirpath, pathlib.Path):
        dirpath = pathlib.Path(dirpath)
    if not dirpath.is_dir():
        raise ValueError("dirpath must be a valid directory.")
    image_filenames = glob_ext(dirpath, ext=ext)
    images: typing.List[mioi.Image] = []
    for fn in image_filenames:
        image = mioi.Image.from_path(fn)
        images.append(image)
    return images


def gather_images_and_masks(
    image_dir: intnormt.PathLike,
    mask_dir: intnormt.PathLike | None = None,
    *,
    ext: builtins.str = "nii*",
) -> typing.Tuple[typing.List[intnormt.Image], typing.Sequence[intnormt.Image | None]]:
    images = gather_images(image_dir, ext=ext)
    if mask_dir is not None:
        masks = gather_images(mask_dir, ext=ext)
    else:
        masks = [None] * len(images)
    return images, masks


def glob_ext(
    dirpath: intnormt.PathLike, *, ext: builtins.str = "nii*"
) -> typing.List[pathlib.Path]:
    """return a sorted list of ext files for a given directory path"""
    if not isinstance(dirpath, pathlib.Path):
        dirpath = pathlib.Path(dirpath)
    assert dirpath.is_dir()
    filenames = sorted(dirpath.resolve().glob(f"*.{ext}"))
    return filenames


def split_filename(
    filepath: intnormt.PathLike,
    /,
    *,
    resolve: builtins.bool = False,
) -> intnormt.SplitFilename:
    """split a filepath into the directory, base, and extension"""
    filepath = pathlib.Path(filepath)
    if resolve:
        filepath = filepath.resolve()
    path = filepath.parent
    _base = pathlib.Path(filepath.stem)
    ext = filepath.suffix
    if ext == ".gz":
        ext2 = _base.suffix
        base = str(_base.stem)
        ext = ext2 + ext
    else:
        base = str(_base)
    return intnormt.SplitFilename(pathlib.Path(path), base, ext)
