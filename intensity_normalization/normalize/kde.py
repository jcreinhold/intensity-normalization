"""Kernel density estimation-based tissue mode normalization
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["KDENormalize"]

import argparse
import builtins
import typing

import intensity_normalization as intnorm
import intensity_normalization.normalize.base as intnormb
import intensity_normalization.typing as intnormt
import intensity_normalization.util.histogram_tools as intnormhisttool


class KDENormalize(intnormb.NormalizeBase):
    """
    use kernel density estimation to find the peak of the white
    matter in the histogram of a (skull-stripped) brain MR image.
    Normalize the WM of the image to norm_value (default = 1.)
    """

    def calculate_location(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> float:
        return 0.0

    def calculate_scale(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> float:
        voi = self._get_voi(image, mask, modality=modality)
        tissue_mode = intnormhisttool.get_tissue_mode(voi, modality=modality)
        return tissue_mode

    @staticmethod
    def name() -> str:
        return "kde"

    @staticmethod
    def fullname() -> str:
        return "Kernel Density Estimation"

    @staticmethod
    def description() -> str:
        desc = "Use kernel density estimation to find the WM mode from "
        desc += "a smoothed histogram and normalize an NIfTI MR image."
        return desc

    @staticmethod
    def get_parent_parser(
        desc: builtins.str,
        valid_modalities: typing.Set[builtins.str] = intnorm.VALID_PEAKS,
    ) -> argparse.ArgumentParser:
        return super(KDENormalize, KDENormalize).get_parent_parser(
            desc, valid_modalities
        )
