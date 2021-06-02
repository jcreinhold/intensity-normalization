# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.nyul

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 02, 2021
"""

__all__ = [
    "NyulNormalize",
]

from typing import List, Optional

import numpy as np
from scipy.interpolate import interp1d

from intensity_normalization.type import Array, ArrayOrNifti, Vector
from intensity_normalization.normalize.base import NormalizeSetBase


class NyulNormalize(NormalizeSetBase):
    def __init__(
        self,
        i_min: float = 1.0,
        i_max: float = 99.0,
        i_s_min: float = 1.0,
        i_s_max: float = 100.0,
        l_percentile: float = 10.0,
        u_percentile: float = 90.0,
        step: float = 10.0,
    ):
        super().__init__()
        self.i_min = i_min
        self.i_max = i_max
        self.i_s_min = i_s_min
        self.i_s_max = i_s_max
        self.l_percentile = l_percentile
        self.u_percentile = u_percentile
        self.step = step

    def normalize_array(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> Array:
        mask = self._get_mask(data, mask, modality)
        masked = data[mask > 0.0]
        landmarks = self.get_landmarks(masked)
        f = interp1d(landmarks, self.standard_scale, fill_value="extrapolate")
        normalized = f(data)
        return normalized

    @property
    def percentiles(self):
        percs = np.arange(self.l_percentile, self.u_percentile + self.step, self.step)
        return np.concatenate(([self.i_min], percs, [self.i_max]))

    def get_landmarks(self, image: Array) -> Vector:
        return np.percentile(image, self.percentiles)

    def fit(
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs
    ):
        """Compute standard scale for piecewise linear histogram matching

        Args:
            images: set of NifTI MR image paths which are to be normalized
            masks: set of corresponding masks (if not provided, estimated)
        """
        n_percs = len(self.percentiles)
        standard_scale = np.zeros(n_percs)
        masks = masks or ([None] * len(images))
        n_images = len(images)
        assert n_images == len(masks)
        for i, (image, mask) in enumerate(zip(images, masks)):
            mask = self._get_mask(image, mask, modality)
            masked = image[mask > 0.0]
            landmarks = self.get_landmarks(masked)
            min_p = np.percentile(masked, self.i_min)
            max_p = np.percentile(masked, self.i_max)
            f = interp1d([min_p, max_p], [self.i_s_min, self.i_s_max])
            landmarks = np.array(f(landmarks))
            standard_scale += landmarks
        self.standard_scale = standard_scale / n_images

    @staticmethod
    def name() -> str:
        return "lsq"
