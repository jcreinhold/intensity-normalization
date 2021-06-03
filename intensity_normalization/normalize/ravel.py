# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.ravel

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 02, 2021
"""

__all__ = [
    "RavelNormalize",
]

import logging
from typing import List, Optional

import nibabel as nib
import numpy as np

from intensity_normalization.type import Array, ArrayOrNifti, Vector
from intensity_normalization.normalize.base import NormalizeSetBase
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.util.tissue_membership import find_tissue_memberships

logger = logging.getLogger(__name__)

try:
    import ants
except (ModuleNotFoundError, ImportError):
    logger.warning("ANTsPy not installed. Install antspyx to use co-registration.")
    raise


class RavelNormalize(NormalizeSetBase):
    def __init__(
        self,
        membership_threshold: float = 0.99,
        smoothness: float = 0.25,
        max_num_control_voxels: int = 10000,
        register: bool = False,
        control_proportion: float = 1.0,
    ):
        super().__init__()
        self.membership_threshold = membership_threshold
        self.smoothness = smoothness
        self.max_num_control_voxels = max_num_control_voxels
        self.register = register
        self.control_proportion = control_proportion

    def calculate_location(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        return 0.0

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> float:
        return 1.0

    def fit(
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError

    @staticmethod
    def name() -> str:
        return "ravel"
