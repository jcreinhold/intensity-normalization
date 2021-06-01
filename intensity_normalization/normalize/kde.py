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


class KDENormalize(NormalizeBase):
    def calculate_location(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        return 0.0

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        modality = self._get_modality(modality)
        voi = self._get_voi(data, mask, modality)
        tissue_mode = get_tissue_mode(voi, modality)
        return tissue_mode
