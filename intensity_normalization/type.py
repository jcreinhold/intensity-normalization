# -*- coding: utf-8 -*-
"""
intensity_normalization.type

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "allowed_interpolators",
    "allowed_orientations",
    "allowed_transforms",
    "ArgType",
    "Array",
    "ArrayOrNifti",
    "interp_type_dict",
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
ArrayOrNifti = Union[np.ndarray, nib.Nifti1Image]
PathLike = Union[str, Path]
Vector = np.ndarray

interp_type_dict = dict(
    linear=0, nearest_neighbor=1, gaussian=2, windowed_sinc=3, bspline=4,
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
