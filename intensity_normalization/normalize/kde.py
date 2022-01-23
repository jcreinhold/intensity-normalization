"""Kernel density estimation-based tissue mode normalization
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["KDENormalize"]

from argparse import ArgumentParser
from typing import Optional, Set

from intensity_normalization import VALID_PEAKS
from intensity_normalization.normalize.base import NormalizeBase
from intensity_normalization.typing import Array
from intensity_normalization.util.histogram_tools import get_tissue_mode


class KDENormalize(NormalizeBase):
    """
    use kernel density estimation to find the peak of the white
    matter in the histogram of a (skull-stripped) brain MR image.
    Normalize the WM of the image to norm_value (default = 1.)
    """

    def calculate_location(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> float:
        return 0.0

    def calculate_scale(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> float:
        modality = self._get_modality(modality)
        voi = self._get_voi(data, mask, modality)
        tissue_mode = get_tissue_mode(voi, modality)
        return tissue_mode

    @staticmethod
    def name() -> str:
        return "kde"

    @staticmethod
    def fullname() -> str:
        return "Kernel Density Estimation"

    @staticmethod
    def description() -> str:
        return (
            "Use kernel density estimation to find the WM mode from "
            "a smoothed histogram and normalize an NIfTI MR image."
        )

    @staticmethod
    def get_parent_parser(
        desc: str,
        valid_modalities: Set[str] = VALID_PEAKS,
    ) -> ArgumentParser:
        return super(KDENormalize, KDENormalize).get_parent_parser(
            desc, valid_modalities
        )
