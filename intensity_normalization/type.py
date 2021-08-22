# -*- coding: utf-8 -*-
"""
intensity_normalization.type

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "allowed_interpolators",
    "allowed_metrics",
    "allowed_orientations",
    "allowed_transforms",
    "ArgType",
    "Array",
    "ArrayOrNifti",
    "dir_path",
    "file_path",
    "new_parse_type",
    "nonnegative_float",
    "nonnegative_int",
    "positive_float",
    "positive_int",
    "positive_int_or_none",
    "positive_odd_int_or_none",
    "probability_float",
    "probability_float_or_none",
    "save_file_path",
    "save_nifti_path",
    "interp_type_dict",
    "NiftiImage",
    "PathLike",
    "Vector",
]

from argparse import ArgumentTypeError, Namespace
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import nibabel as nib
import numpy as np

ArgType = Optional[Union[Namespace, List[str]]]
Array = np.ndarray
NiftiImage = nib.Nifti1Image
ArrayOrNifti = Union[np.ndarray, nib.Nifti1Image]
PathLike = Union[str, Path]
Vector = np.ndarray

interp_type_dict = dict(
    linear=0,
    nearest_neighbor=1,
    gaussian=2,
    windowed_sinc=3,
    bspline=4,
)

# copied from: (https://github.com/ANTsX/ANTsPy/blob/
# 5b4b8273815b681b0542a3dc8846713e2ebb786e/ants/registration/reorient_image.py)
allowed_orientations = frozenset(
    {
        "RIP",
        "LIP",
        "RSP",
        "LSP",
        "RIA",
        "LIA",
        "RSA",
        "LSA",
        "IRP",
        "ILP",
        "SRP",
        "SLP",
        "IRA",
        "ILA",
        "SRA",
        "SLA",
        "RPI",
        "LPI",
        "RAI",
        "LAI",
        "RPS",
        "LPS",
        "RAS",
        "LAS",
        "PRI",
        "PLI",
        "ARI",
        "ALI",
        "PRS",
        "PLS",
        "ARS",
        "ALS",
        "IPR",
        "SPR",
        "IAR",
        "SAR",
        "IPL",
        "SPL",
        "IAL",
        "SAL",
        "PIR",
        "PSR",
        "AIR",
        "ASR",
        "PIL",
        "PSL",
        "AIL",
        "ASL",
    }
)

# copied from (https://github.com/ANTsX/ANTsPy/blob/
# 4474f894d184da98a099cd9c852795c384fa3b8f/ants/registration/interface.py)
allowed_transforms = frozenset(
    {
        "SyNBold",
        "SyNBoldAff",
        "ElasticSyN",
        "Elastic",
        "SyN",
        "SyNRA",
        "SyNOnly",
        "SyNAggro",
        "SyNCC",
        "TRSAA",
        "SyNabp",
        "SyNLessAggro",
        "TV[1]",
        "TV[2]",
        "TV[3]",
        "TV[4]",
        "TV[5]",
        "TV[6]",
        "TV[7]",
        "TV[8]",
        "TVMSQ",
        "TVMSQC",
        "Rigid",
        "Similarity",
        "Translation",
        "Affine",
        "AffineFast",
        "BOLDAffine",
        "QuickRigid",
        "DenseRigid",
        "BOLDRigid",
        "antsRegistrationSyN[r]",
        "antsRegistrationSyN[t]",
        "antsRegistrationSyN[a]",
        "antsRegistrationSyN[b]",
        "antsRegistrationSyN[s]",
        "antsRegistrationSyN[br]",
        "antsRegistrationSyN[sr]",
        "antsRegistrationSyN[bo]",
        "antsRegistrationSyN[so]",
        "antsRegistrationSyNQuick[r]",
        "antsRegistrationSyNQuick[t]",
        "antsRegistrationSyNQuick[a]",
        "antsRegistrationSyNQuick[b]",
        "antsRegistrationSyNQuick[s]",
        "antsRegistrationSyNQuick[br]",
        "antsRegistrationSyNQuick[sr]",
        "antsRegistrationSyNQuick[bo]",
        "antsRegistrationSyNQuick[so]",
        "antsRegistrationSyNRepro[r]",
        "antsRegistrationSyNRepro[t]",
        "antsRegistrationSyNRepro[a]",
        "antsRegistrationSyNRepro[b]",
        "antsRegistrationSyNRepro[s]",
        "antsRegistrationSyNRepro[br]",
        "antsRegistrationSyNRepro[sr]",
        "antsRegistrationSyNRepro[bo]",
        "antsRegistrationSyNRepro[so]",
        "antsRegistrationSyNQuickRepro[r]",
        "antsRegistrationSyNQuickRepro[t]",
        "antsRegistrationSyNQuickRepro[a]",
        "antsRegistrationSyNQuickRepro[b]",
        "antsRegistrationSyNQuickRepro[s]",
        "antsRegistrationSyNQuickRepro[br]",
        "antsRegistrationSyNQuickRepro[sr]",
        "antsRegistrationSyNQuickRepro[bo]",
        "antsRegistrationSyNQuickRepro[so]",
    }
)

# copied from: (https://github.com/ANTsX/ANTsPy/blob/
# 4474f894d184da98a099cd9c852795c384fa3b8f/ants/registration/apply_transforms.py)
allowed_interpolators = frozenset(
    {
        "linear",
        "nearestNeighbor",
        "multiLabel",
        "gaussian",
        "bSpline",
        "cosineWindowedSinc",
        "welchWindowedSinc",
        "hammingWindowedSinc",
        "lanczosWindowedSinc",
        "genericLabel",
    }
)

# copied from : (https://github.com/ANTsX/ANTsPy/blob/
# f2aec7283d26d914d98e2b440e4d2badff78da38/ants/registration/interface.py)
allowed_metrics = frozenset(
    {
        "CC",
        "mattes",
        "meansquares",
        "demons",
    }
)


def return_none(func: Callable) -> Callable:
    def new_func(self, string: Any) -> Any:  # type: ignore[no-untyped-def]
        if string is None:
            return None
        elif isinstance(string, str):
            if string.lower() in ("none", "null"):
                return None
        return func(self, string)

    return new_func


class _ParseType:
    @property
    def __name__(self) -> str:
        name = self.__class__.__name__
        assert isinstance(name, str)
        return name

    def __str__(self) -> str:
        return self.__name__


class save_file_path(_ParseType):
    def __call__(self, string: str) -> Path:
        if not string.isprintable():
            msg = "String must only contain printable characters."
            raise ArgumentTypeError(msg)
        path = Path(string)
        return path


class save_nifti_path(_ParseType):
    def __call__(self, string: str) -> Path:
        not_nifti = not string.endswith(".nii.gz") and not string.endswith(".nii")
        if not_nifti or not string.isprintable():
            msg = (
                f"{string} is not a valid path to a NIfTI file. "
                "Needs to end with .nii or .nii.gz and can "
                "only contain printable characters."
            )
            raise ArgumentTypeError(msg)
        path = Path(string)
        return path


class dir_path(_ParseType):
    def __call__(self, string: str) -> str:
        path = Path(string)
        if not path.is_dir():
            msg = f"{string} is not a valid directory path."
            raise ArgumentTypeError(msg)
        return str(path.resolve())


class file_path(_ParseType):
    def __call__(self, string: str) -> str:
        path = Path(string)
        if not path.is_file():
            msg = f"{string} is not a valid file path."
            raise ArgumentTypeError(msg)
        return str(path)


class positive_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.0:
            msg = f"{string} needs to be a positive float."
            raise ArgumentTypeError(msg)
        return num


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            msg = f"{string} needs to be a positive integer."
            raise ArgumentTypeError(msg)
        return num


class positive_odd_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[int, None]:
        num = int(string)
        if num <= 0 or not (num % 2):
            msg = f"{string} needs to be a positive odd integer."
            raise ArgumentTypeError(msg)
        return num


class positive_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[int, None]:
        return positive_int()(string)


class nonnegative_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num < 0:
            msg = f"{string} needs to be a nonnegative integer."
            raise ArgumentTypeError(msg)
        return num


class nonnegative_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0:
            msg = f"{string} needs to be a nonnegative float."
            raise ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0 or num > 1.0:
            msg = f"{string} needs to be between 0 and 1."
            raise ArgumentTypeError(msg)
        return num


class probability_float_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[float, None]:
        return probability_float()(string)


class NewParseType:
    def __init__(self, func: Callable, name: str):
        self.name = name
        self.func = func

    def __str__(self) -> str:
        return self.name

    def __call__(self, val: Any) -> Any:
        return self.func(val)


def new_parse_type(func: Callable, name: str) -> NewParseType:
    return NewParseType(func, name)
