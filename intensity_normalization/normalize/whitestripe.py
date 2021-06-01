#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity-normalization.normalize.fcm

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = []

from typing import Optional

import numpy as np

from intensity_normalization.type import Array
from intensity_normalization.normalize.base import NormalizeBase
from intensity_normalization.util.histogram_tools import get_tissue_mode


class WhiteStripeNormalize(NormalizeBase):
    def __init__(
        self,
        width: float = 0.05,
        width_l: Optional[float] = None,
        width_u: Optional[float] = None,
    ):
        super().__init__()
        self.width = width
        self.width_l = width_l or width
        self.width_u = width_u or width

    def calculate_location(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        return data[self.whitestripe].mean()

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        return data[self.whitestripe].std()

    def setup(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ):
        voi = self._get_voi(data, mask, modality)
        wm_mode = get_tissue_mode(voi, modality)
        wm_mode_quantile = np.mean(voi < wm_mode)
        lower_bound = max(wm_mode_quantile - self.width_l, 0)
        upper_bound = min(wm_mode_quantile + self.width_u, 1)
        ws_l, ws_u = np.quantile(voi, (lower_bound, upper_bound))
        self.whitestripe = (voi > ws_l) & (voi < ws_u)

    def teardown(self):
        del self.whitestripe
