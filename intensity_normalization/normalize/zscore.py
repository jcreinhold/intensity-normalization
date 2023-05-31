"""Z-score normalize image (voxel-wise subtract mean, divide by standard deviation)
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["ZScoreNormalize"]

import argparse
import typing

import intensity_normalization.errors as intnorme
import intensity_normalization.normalize.base as intnormb
import intensity_normalization.typing as intnormt


class ZScoreNormalize(intnormb.LocationScaleCLIMixin, intnormb.SingleImageNormalizeCLI):
    def __init__(self, *, norm_value: float = 1.0, **kwargs: typing.Any):
        """Voxel-wise subtract the mean and divide by the standard deviation."""
        super().__init__(norm_value=norm_value, **kwargs)
        self.voi: intnormt.ImageLike | None = None

    def calculate_location(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> float:
        if self.voi is None:
            raise intnorme.NormalizationError("'voi' needs to be set.")
        loc: float = float(self.voi.mean())
        return loc

    def calculate_scale(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> float:
        if self.voi is None:
            raise intnorme.NormalizationError("'voi' needs to be set.")
        scale: float = float(self.voi.std())
        return scale

    def setup(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> None:
        self.voi = self._get_voi(image, mask, modality=modality)

    def teardown(self) -> None:
        del self.voi
        self.voi = None

    @staticmethod
    def name() -> str:
        return "zscore"

    @staticmethod
    def fullname() -> str:
        return "Z-Score"

    @staticmethod
    def description() -> str:
        return "Standardize an MR image by the foreground intensities."

    def plot_histogram_from_args(
        self,
        args: argparse.Namespace,
        /,
        normalized: intnormt.ImageLike,
        mask: intnormt.ImageLike | None = None,
    ) -> None:
        if mask is None:
            mask = self.estimate_foreground(normalized)
        super().plot_histogram_from_args(args, normalized, mask)
