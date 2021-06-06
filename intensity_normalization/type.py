# -*- coding: utf-8 -*-
"""
intensity_normalization.type

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "ArgType",
    "Array",
    "ArrayOrNifti",
    "NiftiImage",
    "PathLike",
    "Vector",
]

from argparse import Namespace
from typing import List, Optional, Union

import nibabel as nib
import numpy as np
from pathlib import Path

ArgType = Optional[Union[Namespace, List[str]]]
Array = np.ndarray
NiftiImage = nib.Nifti1Image
ArrayOrNifti = Union[Array, NiftiImage]
PathLike = Union[str, Path]
Vector = np.ndarray
