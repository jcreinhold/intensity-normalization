#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.io

assortment of input/output utilities for the project

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 24, 2018
"""

from pathlib import Path
from typing import Tuple, Union


def split_filename(filepath: Union[str, Path]) -> Tuple[Path, str, str]:
    """ split a filepath into the directory, base, and extension """
    filepath = Path(filepath).resolve()
    path = filepath.parent
    base = Path(filepath.stem)
    ext = filepath.suffix
    if ext == ".gz":
        ext2 = base.suffix
        base = base.stem
        ext = ext2 + ext
    return Path(path), str(base), ext


def glob_ext(dirpath: Union[str, Path], ext: str = "nii*"):
    """ return a sorted list of ext files for a given directory path """
    if isinstance(dirpath, str):
        dirpath = Path(dirpath)
    filenames = sorted(dirpath.resolve().glob(f"*.{ext}"))
    return filenames
