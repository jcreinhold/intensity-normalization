"""Kernel density estimation-based tissue mode normalization
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["KDENormalize"]

import intensity_normalization.normalize.base as intnormb
import intensity_normalization.typing as intnormt
import intensity_normalization.util.histogram_tools as intnormhisttool


class KDENormalize(intnormb.LocationScaleCLIMixin, intnormb.SingleImageNormalizeCLI):
    """
    Use kernel density estimation to fit a smoothed histogram of intensities
    of a (skull-stripped) brain MR image, then find the peak of the white
    matter (by default) in the smoothed histogram. Finally, normalize the
    white matter mode of the image to norm_value (default = 1.)
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
