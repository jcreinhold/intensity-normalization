# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.whitestripe

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "WhiteStripeNormalize",
]

from argparse import ArgumentParser
from typing import Optional

import numpy as np

from intensity_normalization.type import Array
from intensity_normalization.normalize.base import NormalizeBase
from intensity_normalization.util.histogram_tools import get_tissue_mode


class WhiteStripeNormalize(NormalizeBase):
    """
    find the normal appearing white matter of the input MR image and
    use those values to standardize the data (i.e., subtract the mean of
    the values in the indices and divide by the std of those values)
    """

    def __init__(
        self,
        width: float = 0.05,
        width_l: Optional[float] = None,
        width_u: Optional[float] = None,
    ):
        super().__init__()
        self.width_l = width_l or width
        self.width_u = width_u or width

    def calculate_location(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        loc: float = data[self.whitestripe].mean()
        return loc

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        scale: float = data[self.whitestripe].std()
        return scale

    def setup(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> None:
        if modality is None:
            modality = "t1"
        mask = self._get_mask(data, mask, modality)
        masked = data * mask
        voi = data[mask]
        wm_mode = get_tissue_mode(voi, modality)
        wm_mode_quantile = np.mean(voi < wm_mode)
        lower_bound = max(wm_mode_quantile - self.width_l, 0)
        upper_bound = min(wm_mode_quantile + self.width_u, 1)
        ws_l, ws_u = np.quantile(voi, (lower_bound, upper_bound))
        self.whitestripe = (masked > ws_l) & (masked < ws_u)

    def teardown(self) -> None:
        del self.whitestripe

    @staticmethod
    def name() -> str:
        return "ws"

    @staticmethod
    def description() -> str:
        return "Standardize the normal appearing WM of a NIfTI MR image."

    @staticmethod
    def add_method_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Method")
        parser.add_argument(
            "--width", default=0.05, type=float, help="width of the whitestripe",
        )
        return parent_parser
