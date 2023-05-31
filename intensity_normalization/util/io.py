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
    "zip_with_nones",
]

import collections.abc
import pathlib
import typing

import pymedio.image as mioi

import intensity_normalization.typing as intnormt

PymedioImageList = list[mioi.Image]
PymedioMaskListOrNone = typing.Union[PymedioImageList, None]


def gather_images(
    dirpath: intnormt.PathLike,
    *,
    ext: str = "nii*",
    exclude: collections.abc.Sequence[str] = (),
) -> PymedioImageList:
    """return all images of extension `ext` from a directory"""
    if not isinstance(dirpath, pathlib.Path):
        dirpath = pathlib.Path(dirpath)
    if not dirpath.is_dir():
        raise ValueError("'dirpath' must be a valid directory.")
    image_filenames = glob_ext(dirpath, ext=ext, exclude=exclude)
    images: PymedioImageList = []
    for fn in image_filenames:
        image: mioi.Image = mioi.Image.from_path(fn)
        images.append(image)
    return images


def gather_images_and_masks(
    image_dir: intnormt.PathLike,
    mask_dir: intnormt.PathLike | None = None,
    *,
    ext: str = "nii*",
    exclude: collections.abc.Sequence[str] = (),
) -> tuple[PymedioImageList, PymedioMaskListOrNone]:
    images = gather_images(image_dir, ext=ext, exclude=exclude)
    masks: PymedioMaskListOrNone
    if mask_dir is not None:
        masks = gather_images(mask_dir, ext=ext, exclude=exclude)
    else:
        masks = None
    return images, masks


def glob_ext(
    dirpath: intnormt.PathLike,
    *,
    ext: str = "nii*",
    exclude: collections.abc.Sequence[str] = (),
) -> list[pathlib.Path]:
    """return a sorted list of ext files for a given directory path"""
    dirpath = pathlib.Path(dirpath)
    if not dirpath.is_dir():
        raise ValueError("'dirpath' must be a directory.")
    filenames = sorted(
        dp
        for dp in dirpath.resolve().glob(f"*.{ext}")
        if all(exc not in str(dp) for exc in exclude)
    )
    return filenames


def split_filename(
    filepath: intnormt.PathLike,
    /,
    *,
    resolve: bool = False,
) -> intnormt.SplitFilename:
    """split a filepath into the directory, base, and extension
    Examples:
        >>> split_filename("path/base.ext")
        SplitFilename(path=PosixPath('path'), base='base', ext='.ext')
    """
    if not str(filepath):
        raise ValueError("filepath must be a non-empty string.")
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


Zipped = typing.Generator[tuple[typing.Any, ...], None, None]


def zip_with_nones(*args: typing.Sequence[typing.Any] | None) -> Zipped:
    """zip sequence args but if an arg is None, yield None in that argument index
    Examples:
        >>> for x, y, z in zip_with_nones((1, 2), None, ("a", "b")):
        ...    print(x, y, z)
        1 None a
        2 None b
    """
    _args: list[typing.Any] = list(args)
    none_indices: list[int] = []
    length: int | None = None
    for i, seq_or_none in enumerate(args):
        try:
            _length = len(seq_or_none)  # type: ignore[arg-type]
        except TypeError:
            if seq_or_none is not None:
                raise RuntimeError("Only sequences or 'None' allowed.")
            none_indices.append(i)
        else:
            if length is None:
                length = _length
            elif length is not None and length != _length:
                raise RuntimeError("All sequences should be the same length.")

    def nones(length: int) -> typing.Generator[None, None, None]:
        for _ in range(length):
            yield None

    if length is None:
        raise RuntimeError("At least one argument needs to be a sequence.")

    for idx in none_indices:
        _args[idx] = nones(length)

    return typing.cast(Zipped, zip(*_args))
