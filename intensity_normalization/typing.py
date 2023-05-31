"""Project-specific types
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = [
    "allowed_interpolators",
    "allowed_metrics",
    "allowed_orientations",
    "allowed_transforms",
    "ArgType",
    "dir_path",
    "file_path",
    "ImageLike",
    "interp_type_dict",
    "Modality",
    "new_parse_type",
    "nonnegative_float",
    "nonnegative_int",
    "PathLike",
    "positive_float",
    "positive_int",
    "positive_int_or_none",
    "positive_odd_int_or_none",
    "probability_float",
    "probability_float_or_none",
    "save_file_path",
    "SplitFilename",
    "TissueType",
]

import argparse
import collections.abc
import enum
import os
import pathlib
import typing

import numpy as np
import numpy.typing as npt

import intensity_normalization as intnorm

ArgType = typing.Union[argparse.Namespace, list[str], None]
PathLike = typing.Union[str, os.PathLike]
ShapeLike = typing.Union[
    typing.SupportsIndex, collections.abc.Sequence[typing.SupportsIndex]
]

_MODALITIES = [(vm.upper(), vm) for vm in sorted(intnorm.VALID_MODALITIES)]


class Modality(enum.Enum):
    FLAIR: str = "flair"
    MD: str = "md"
    OTHER: str = "other"
    PD: str = "pd"
    T1: str = "t1"
    T2: str = "t2"

    @classmethod
    def from_string(cls: typing.Type, string: str | Modality) -> Modality:
        if isinstance(string, cls):
            modality: Modality = string
            return modality
        for name, value in _MODALITIES:
            if string == value:
                modality = getattr(cls, name)
                return modality
        msg = f"'string' must be one of {intnorm.VALID_MODALITIES}. Got '{string}'."
        raise ValueError(msg)


# not ideal DRY, but avoid functional enum API for better IDE support & flake8
if set(m.value for m in Modality) != set(intnorm.VALID_MODALITIES):
    raise RuntimeError("Modalities enum out of sync with VALID_MODALITIES.")


class TissueType(enum.Enum):
    CSF: str = "csf"
    GM: str = "gm"
    WM: str = "wm"

    @classmethod
    def from_string(cls, string: str) -> TissueType:
        if string.lower() == "csf":
            return TissueType.CSF
        elif string.lower() == "gm":
            return TissueType.GM
        elif string.lower() == "wm":
            return TissueType.WM
        else:
            raise ValueError(f"'string' must be 'csf', 'gm', or 'wm'. Got '{string}'.")

    def to_int(self) -> int:
        if self == TissueType.CSF:
            return 0
        elif self == TissueType.GM:
            return 1
        elif self == TissueType.WM:
            return 2
        else:
            raise ValueError("Unexpected enum.")

    def to_fullname(self) -> str:
        if self == TissueType.CSF:
            return "Cerebrospinal fluid"
        elif self == TissueType.GM:
            return "Grey matter"
        elif self == TissueType.WM:
            return "White matter"
        else:
            raise ValueError("Unexpected enum.")


class SplitFilename(typing.NamedTuple):
    path: pathlib.Path
    base: str
    ext: str


interp_type_dict = dict(
    linear=0,
    nearest_neighbor=1,
    gaussian=2,
    windowed_sinc=3,
    bspline=4,
)

# copied from:
# https://github.com/ANTsX/ANTsPy/blob/5b4b8273815b681b0542a3dc8846713e2ebb786e/ants/registration/reorient_image.py
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

# copied from:
# https://github.com/ANTsX/ANTsPy/blob/4474f894d184da98a099cd9c852795c384fa3b8f/ants/registration/interface.py
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

# copied from:
# https://github.com/ANTsX/ANTsPy/blob/4474f894d184da98a099cd9c852795c384fa3b8f/ants/registration/apply_transforms.py
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

# copied from:
# https://github.com/ANTsX/ANTsPy/blob/f2aec7283d26d914d98e2b440e4d2badff78da38/ants/registration/interface.py
allowed_metrics = frozenset(
    {
        "CC",
        "mattes",
        "meansquares",
        "demons",
    }
)


def return_none(
    func: typing.Callable[[typing.Any, typing.Any], typing.Any]
) -> typing.Callable[[typing.Any, typing.Any], typing.Any]:
    def new_func(self: object, string: typing.Any) -> typing.Any:
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
    def __call__(self, string: str) -> pathlib.Path:
        if not string.isprintable():
            msg = f"'{string}' must only contain printable characters."
            raise argparse.ArgumentTypeError(msg)
        path = pathlib.Path(string)
        return path


class dir_path(_ParseType):
    def __call__(self, string: str) -> str:
        path = pathlib.Path(string)
        if not path.is_dir():
            msg = f"'{string}' is not a valid directory path."
            raise argparse.ArgumentTypeError(msg)
        return str(path.resolve())


class file_path(_ParseType):
    def __call__(self, string: str) -> str:
        path = pathlib.Path(string)
        if not path.is_file():
            msg = f"'{string}' is not a valid file path."
            raise argparse.ArgumentTypeError(msg)
        return str(path)


class positive_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.0:
            msg = f"'{string}' needs to be a positive float."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            msg = f"'{string}' needs to be a positive integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_odd_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> int | None:
        num = int(string)
        if num <= 0 or not (num % 2):
            msg = f"'{string}' needs to be a positive odd integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> int | None:
        return positive_int()(string)


class nonnegative_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num < 0:
            msg = f"'{string}' needs to be a non-negative integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class nonnegative_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0:
            msg = f"'{string}' needs to be a non-negative float."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0 or num > 1.0:
            msg = f"'{string}' needs to be between 0 and 1."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> float | None:
        return probability_float()(string)


class NewParseType:
    def __init__(self, func: typing.Callable[[typing.Any], typing.Any], name: str):
        self.name = name
        self.func = func

    def __str__(self) -> str:
        return self.name

    def __call__(self, val: typing.Any) -> typing.Any:
        return self.func(val)


def new_parse_type(
    func: typing.Callable[[typing.Any], typing.Any], name: str
) -> NewParseType:
    return NewParseType(func, name)


S_co = typing.TypeVar("S_co", bound="ImageLike", covariant=True)
T_co = typing.TypeVar("T_co", bound="ImageLike", covariant=True)
U_co = typing.TypeVar("U_co", bound="ImageLike", covariant=True)

NBit = typing.TypeVar("NBit", bound=npt.NBitBase)
Float = typing.Union[np.floating[NBit], float]
Int = typing.Union[np.integer[NBit], int]


class ImageLike(typing.Protocol[S_co, T_co, U_co]):
    """support anything that implements the methods here"""

    def __gt__(self: T_co, other: typing.Any) -> U_co:
        ...

    def __ge__(self: T_co, other: typing.Any) -> U_co:
        ...

    def __lt__(self: T_co, other: typing.Any) -> U_co:
        ...

    def __le__(self: T_co, other: typing.Any) -> U_co:
        ...

    def __and__(self: T_co, other: typing.Any) -> U_co:
        ...

    def __or__(self: T_co, other: typing.Any) -> U_co:
        ...

    def __add__(self: T_co, other: typing.Any) -> S_co:
        ...

    def __sub__(self: T_co, other: typing.Any) -> S_co:
        ...

    def __mul__(self: T_co, other: typing.Any) -> S_co:
        ...

    def __truediv__(self: T_co, other: typing.Any) -> S_co:
        ...

    def __getitem__(self: T_co, item: typing.Any) -> typing.Any:
        ...

    def __iter__(self: T_co) -> T_co:
        ...

    def __array__(self) -> npt.NDArray:
        ...

    def sum(self) -> Float | Int:
        ...

    @property
    def ndim(self) -> Int:
        ...

    def any(
        self,
        axis: int | tuple[int, ...] | None = None,
    ) -> typing.Any:
        ...

    def nonzero(self) -> typing.Any:
        ...

    def squeeze(self) -> typing.Any:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    def mean(self) -> float:
        ...

    def std(self) -> float:
        ...

    def min(self) -> float:
        ...

    def flatten(self: T_co) -> T_co:
        ...

    def reshape(
        self: T_co,
        *shape: typing.SupportsIndex,
        order: typing.Literal["A", "C", "F"] | None = ...,
    ) -> T_co:
        ...

    def transpose(self: T_co, *axes: int) -> T_co:
        ...
