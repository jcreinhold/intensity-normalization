# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.fcm

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "FCMNormalize",
]

from typing import Optional

import numpy as np

from intensity_normalization.type import Array
from intensity_normalization.normalize.base import NormalizeBase
from intensity_normalization.util.tissue_membership import find_tissue_memberships


class FCMNormalize(NormalizeBase):
    """
    Use fuzzy c-means-generated tissue membership (found on a T1-w
    image) to normalize the tissue to norm_value (default = 1.)
    """

    tissue_to_int = {"csf": 0, "gm": 1, "wm": 2}

    def __init__(self, norm_value: float = 1.0, tissue_type: str = "wm"):
        super().__init__(norm_value)
        self.tissue_membership = None
        self.tissue_type = tissue_type

    def calculate_location(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        return 0.0

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        modality = self._get_modality(modality)
        if modality == "t1":
            if mask is None:
                mask = data > 0.0
            tissue_memberships = find_tissue_memberships(data, mask)
            self.tissue_membership = tissue_memberships[
                ..., self.tissue_to_int[self.tissue_type]
            ]
            tissue_mean = np.average(data, weights=self.tissue_membership)
        elif modality != "t1" and mask is None and self.tissue_membership is not None:
            tissue_mean = np.average(data, weights=self.tissue_membership)
        elif modality != "t1" and mask is not None:
            tissue_mean = np.average(data, weights=mask)
        else:
            msg = (
                "Either a T1-w image must be passed to initialize a tissue "
                "membership mask or the tissue memberships must be provided."
            )
            raise ValueError(msg)
        return tissue_mean

    @staticmethod
    def name() -> str:
        return "fcm"
