# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.nyul

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 02, 2021
"""

__all__ = [
    "NyulNormalize",
]

from argparse import Namespace
from typing import List, Optional

import numpy as np
from scipy.interpolate import interp1d

from intensity_normalization.type import Array, ArrayOrNifti, PathLike, Vector
from intensity_normalization.normalize.base import NormalizeSetBase
from intensity_normalization.util.io import glob_ext


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
        voi = self._get_voi(data, mask, modality)
        landmarks = self.get_landmarks(voi)
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
        assert len(images) > 0
        if hasattr(images[0], "get_fdata"):
            images = [image.get_fdata() for image in images]
        if hasattr(masks[0], "get_fdata"):
            masks = [mask.get_fdata() for mask in masks]
        n_percs = len(self.percentiles)
        standard_scale = np.zeros(n_percs)
        masks = masks or ([None] * len(images))
        n_images = len(images)
        assert n_images == len(masks)
        for i, (image, mask) in enumerate(zip(images, masks)):
            voi = self._get_voi(image, mask, modality)
            landmarks = self.get_landmarks(voi)
            min_p = np.percentile(voi, self.i_min)
            max_p = np.percentile(voi, self.i_max)
            f = interp1d([min_p, max_p], [self.i_s_min, self.i_s_max])
            landmarks = np.array(f(landmarks))
            standard_scale += landmarks
        self.standard_scale = standard_scale / n_images

    def save_standard_histogram(self, filename: PathLike):
        assert str(filename).en
        np.save(filename, np.vstack((self.standard_scale, self.percentiles)))

    @staticmethod
    def name() -> str:
        return "lsq"

    @staticmethod
    def description() -> str:
        return (
            "Perform piecewise-linear histogram matching per "
            "Nyul and Udupa given a set of NIfTI MR images."
        )

    @classmethod
    def from_argparse_args(cls, args: Namespace):
        return cls()

    def call_from_argparse_args(self, args: Namespace):
        normalized = self.fit_from_directories(
            args.image_dir, args.mask_dir, return_normalized=True,
        )
        image_filenames = glob_ext(args.image_dir)
        output_filenames = [
            self.append_name_to_file(fn, args.output_dir) for fn in image_filenames
        ]
        for norm_image, fn in zip(normalized, output_filenames):
            norm_image.to_filename(fn)
