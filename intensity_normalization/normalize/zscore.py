# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.zscore

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "ZScoreNormalize",
]

from argparse import Namespace
from typing import Optional

import nibabel as nib

from intensity_normalization.normalize.base import NormalizeBase
from intensity_normalization.type import Array, NiftiImage


class ZScoreNormalize(NormalizeBase):
    def calculate_location(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> float:
        loc: float = self.voi.mean().item()
        return loc

    def calculate_scale(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> float:
        scale: float = self.voi.std().item()
        return scale

    def setup(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> None:
        self.voi = self._get_voi(data, mask, modality)

    def teardown(self) -> None:
        del self.voi

    @staticmethod
    def name() -> str:
        return "zscore"

    @staticmethod
    def fullname() -> str:
        return "Z-Score"

    @staticmethod
    def description() -> str:
        return "Standardize an MR image by the foreground intensities."

    def plot_histogram(
        self,
        args: Namespace,
        normalized: NiftiImage,
        mask: Optional[NiftiImage] = None,
    ) -> None:
        if mask is None:
            mask_data = self.estimate_foreground(normalized.get_fdata())
            mask = nib.Nifti1Image(mask_data, normalized.affine, normalized.header)
        super().plot_histogram(args, normalized, mask)
