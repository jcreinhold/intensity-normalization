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
from intensity_normalization.util.tissue_mask import find_tissue_memberships


class ZScoreNormalize(NormalizeBase):
    def __init__(self):
        super().__init__()

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
