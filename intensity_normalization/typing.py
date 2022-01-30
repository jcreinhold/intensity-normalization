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
    "Image",
    "interp_type_dict",
    "Modalities",
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
    "save_nifti_path",
    "SplitFilename",
    "TissueTypes",
]

import argparse
import builtins
import dataclasses
import enum
import os
import pathlib
import typing

import numpy.typing as npt

import intensity_normalization as intnorm

ArgType = typing.Optional[typing.Union[argparse.Namespace, typing.List[builtins.str]]]
PathLike = typing.Union[builtins.str, os.PathLike]

_MODALITIES = [(vm.upper(), vm) for vm in sorted(intnorm.VALID_MODALITIES)]

Modalities = enum.Enum("Modalities", _MODALITIES, module=__name__)  # type: ignore[misc]


def _modality_from_string(
    cls: typing.Type, string: builtins.str | Modalities
) -> Modalities:
    if isinstance(string, cls):
        modality: Modalities = string
        return modality
    for name, value in _MODALITIES:
        if string == value:
            modality = getattr(cls, name)
            return modality
    msg = f"string must be one of {intnorm.VALID_MODALITIES}. Got {string}"
    raise ValueError(msg)


Modalities.from_string = classmethod(_modality_from_string)  # type: ignore[attr-defined]


class TissueTypes(enum.Enum):
    CSF: builtins.str = "csf"
    GM: builtins.str = "gm"
    WM: builtins.str = "wm"

    @classmethod
    def from_string(cls, string: builtins.str) -> TissueTypes:
        if string.lower() == "csf":
            return TissueTypes.CSF
        elif string.lower() == "gm":
            return TissueTypes.GM
        elif string.lower() == "wm":
            return TissueTypes.WM
        else:
            raise ValueError(f"string must be 'csf', 'gm', or 'wm'. Got {string}")

    def to_int(self) -> builtins.int:
        if self == TissueTypes.CSF:
            return 0
        elif self == TissueTypes.GM:
            return 1
        elif self == TissueTypes.WM:
            return 2
        else:
            raise ValueError("Unexpected enum.")

    def to_fullname(self) -> builtins.str:
        if self == TissueTypes.CSF:
            return "Cerebrospinal fluid"
        elif self == TissueTypes.GM:
            return "Grey matter"
        elif self == TissueTypes.WM:
            return "White matter"
        else:
            raise ValueError("Unexpected enum.")


@dataclasses.dataclass(frozen=True)
class SplitFilename:
    path: pathlib.Path
    base: builtins.str
    ext: builtins.str

    def __iter__(self) -> typing.Iterator[typing.Any]:
        return iter(dataclasses.astuple(self))


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


def return_none(
    func: typing.Callable[[typing.Any, typing.Any], typing.Any]
) -> typing.Callable[[typing.Any, typing.Any], typing.Any]:
    def new_func(self: builtins.object, string: typing.Any) -> typing.Any:
        if string is None:
            return None
        elif isinstance(string, str):
            if string.lower() in ("none", "null"):
                return None
        return func(self, string)

    return new_func


class _ParseType:
    @property
    def __name__(self) -> builtins.str:
        name = self.__class__.__name__
        assert isinstance(name, str)
        return name

    def __str__(self) -> builtins.str:
        return self.__name__


class save_file_path(_ParseType):
    def __call__(self, string: builtins.str) -> pathlib.Path:
        if not string.isprintable():
            msg = "String must only contain printable characters."
            raise argparse.ArgumentTypeError(msg)
        path = pathlib.Path(string)
        return path


class save_nifti_path(_ParseType):
    def __call__(self, string: builtins.str) -> pathlib.Path:
        not_nifti = not string.endswith(".nii.gz") and not string.endswith(".nii")
        if not_nifti or not string.isprintable():
            msg = (
                f"{string} is not a valid path to a NIfTI file. "
                "Needs to end with .nii or .nii.gz and can "
                "only contain printable characters."
            )
            raise argparse.ArgumentTypeError(msg)
        path = pathlib.Path(string)
        return path


class dir_path(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.str:
        path = pathlib.Path(string)
        if not path.is_dir():
            msg = f"{string} is not a valid directory path."
            raise argparse.ArgumentTypeError(msg)
        return str(path.resolve())


class file_path(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.str:
        path = pathlib.Path(string)
        if not path.is_file():
            msg = f"{string} is not a valid file path."
            raise argparse.ArgumentTypeError(msg)
        return str(path)


class positive_float(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.float:
        num = float(string)
        if num <= 0.0:
            msg = f"{string} needs to be a positive float."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.int:
        num = int(string)
        if num <= 0:
            msg = f"{string} needs to be a positive integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_odd_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: builtins.str) -> builtins.int | None:
        num = int(string)
        if num <= 0 or not (num % 2):
            msg = f"{string} needs to be a positive odd integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: builtins.str) -> builtins.int | None:
        return positive_int()(string)


class nonnegative_int(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.int:
        num = int(string)
        if num < 0:
            msg = f"{string} needs to be a nonnegative integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class nonnegative_float(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.float:
        num = float(string)
        if num < 0.0:
            msg = f"{string} needs to be a nonnegative float."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.float:
        num = float(string)
        if num < 0.0 or num > 1.0:
            msg = f"{string} needs to be between 0 and 1."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float_or_none(_ParseType):
    @return_none
    def __call__(self, string: builtins.str) -> builtins.float | None:
        return probability_float()(string)


class NewParseType:
    def __init__(
        self, func: typing.Callable[[typing.Any], typing.Any], name: builtins.str
    ):
        self.name = name
        self.func = func

    def __str__(self) -> str:
        return self.name

    def __call__(self, val: typing.Any) -> typing.Any:
        return self.func(val)


def new_parse_type(
    func: typing.Callable[[typing.Any], typing.Any], name: builtins.str
) -> NewParseType:
    return NewParseType(func, name)


class Image(typing.Protocol):
    """support anything that implements the methods here"""

    def __gt__(self, other: typing.Any) -> typing.Any:
        ...

    def __ge__(self, other: typing.Any) -> typing.Any:
        ...

    def __lt__(self, other: typing.Any) -> typing.Any:
        ...

    def __le__(self, other: typing.Any) -> typing.Any:
        ...

    def __and__(self, other: Image) -> Image:
        ...

    def __or__(self, other: Image) -> Image:
        ...

    def __mul__(self, other: Image) -> Image:
        ...

    def __truediv__(self, other: Image) -> Image:
        ...

    def __getitem__(
        self, item: typing.Tuple[builtins.slice, ...] | builtins.int | Image
    ) -> typing.Any:
        ...

    def sum(self) -> builtins.float:
        ...

    @property
    def ndim(self) -> builtins.int:
        ...

    def any(
        self,
        axis: builtins.int | typing.Tuple[builtins.int, ...] | None = None,
    ) -> typing.Any:
        ...

    def nonzero(self) -> typing.Any:
        ...

    def squeeze(self) -> typing.Any:
        ...

    @property
    def shape(self) -> typing.Tuple[builtins.int, ...]:
        ...

    def mean(self) -> builtins.float:
        ...

    def std(self) -> builtins.float:
        ...

    def min(self) -> builtins.float:
        ...

    def flatten(self) -> Image:
        ...

    @property
    def affine(self) -> npt.NDArray:
        ...

    def reshape(self, shape: typing.Tuple[builtins.int, ...]) -> Image:
        ...
