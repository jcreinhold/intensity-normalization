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

try:
    import ants
except (ModuleNotFoundError, ImportError):
    logging.warning("ANTsPy not installed. Install antspyx to use RAVEL.")
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

    @staticmethod
    def _ravel_correction(control_voxels, unwanted_factors):
        """Correct control voxels by removing trend from unwanted factors

        Args:
            control_voxels (np.ndarray): rows are voxels, columns are images
                (see V matrix in the paper)
            unwanted_factors (np.ndarray): unwanted factors
                (see Z matrix in the paper)

        Returns:
            normalized (np.ndarray): normalized images
        """
        gamma = np.linalg.solve(unwanted_factors, control_voxels)
        fitted = (unwanted_factors @ gamma).T
        residuals = control_voxels - fitted
        voxel_means = np.mean(control_voxels, axis=1, keepdims=True)
        normalized = residuals + voxel_means
        return normalized

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

    @staticmethod
    def description() -> str:
        return (
            "Perform WhiteStripe and then correct for technical "
            "variation with RAVEL on a set of NIfTI MR images."
        )
