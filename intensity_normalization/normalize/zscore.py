# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.zscore

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "ZScoreNormalize",
]

from typing import Optional

from intensity_normalization.type import Array
from intensity_normalization.normalize.base import NormalizeBase


class ZScoreNormalize(NormalizeBase):
    def calculate_location(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        loc: float = self.voi.mean()
        return loc

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        scale: float = self.voi.std()
        return scale

    def setup(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ) -> None:
        self.voi = self._get_voi(data, mask, modality)

    def teardown(self) -> None:
        del self.voi

    @staticmethod
    def name() -> str:
        return "zscore"

    @staticmethod
    def description() -> str:
        return "Standardize an MR image by the foreground intensities."
