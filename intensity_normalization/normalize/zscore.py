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
        return self.voi.mean()

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        return self.voi.std()

    def setup(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ):
        self.voi = self._get_voi(data, mask, modality)

    def teardown(self):
        del self.voi

    @staticmethod
    def name() -> str:
        return "zscore"

    @staticmethod
    def description() -> str:
        return "Standardize an MR image by the foreground intensities."
