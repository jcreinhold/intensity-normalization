# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.nyul

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 02, 2021
"""

__all__ = [
    "NyulNormalize",
]

from argparse import ArgumentParser, Namespace
from typing import List, Optional

import numpy as np
from scipy.interpolate import interp1d

from intensity_normalization.normalize.base import NormalizeFitBase
from intensity_normalization.type import (
    Array,
    ArrayOrNifti,
    PathLike,
    Vector,
    file_path,
    save_file_path,
)


class NyulNormalize(NormalizeFitBase):
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
        self._percentiles = None

    def normalize_array(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> Array:
        voi = self._get_voi(data, mask, modality)
        landmarks = self.get_landmarks(voi)
        f = interp1d(landmarks, self.standard_scale, fill_value="extrapolate")
        normalized: Array = f(data)
        return normalized

    @property
    def percentiles(self) -> Vector:
        if self._percentiles is None:
            percs = np.arange(
                self.l_percentile,
                self.u_percentile + self.step,
                self.step,
            )
            self._percentiles: Vector = np.concatenate(  # type: ignore[no-redef]
                ([self.i_min], percs, [self.i_max])
            )
        return self._percentiles

    def get_landmarks(self, image: Array) -> Vector:
        landmarks: Vector = np.percentile(image, self.percentiles)
        return landmarks

    def _fit(  # type: ignore[no-untyped-def]
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> None:
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
            voi = self._get_voi(image, mask, modality)
            landmarks = self.get_landmarks(voi)
            min_p = np.percentile(voi, self.i_min)
            max_p = np.percentile(voi, self.i_max)
            f = interp1d([min_p, max_p], [self.i_s_min, self.i_s_max])
            landmarks = np.array(f(landmarks))
            standard_scale += landmarks
        self.standard_scale = standard_scale / n_images

    def save_additional_info(  # type: ignore[no-untyped-def]
        self,
        args: Namespace,
        **kwargs,
    ) -> None:
        if args.save_standard_histogram is not None:
            self.save_standard_histogram(args.save_standard_histogram)

    def save_standard_histogram(self, filename: PathLike) -> None:
        np.save(filename, np.vstack((self.standard_scale, self.percentiles)))

    def load_standard_histogram(self, filename: PathLike) -> None:
        data = np.load(filename)
        self.standard_scale = data[0, :]
        self._percentiles = data[1, :]

    @staticmethod
    def name() -> str:
        return "nyul"

    @staticmethod
    def fullname() -> str:
        return "Nyul & Udupa"

    @staticmethod
    def description() -> str:
        return (
            "Perform piecewise-linear histogram matching per "
            "Nyul and Udupa given a set of NIfTI MR images."
        )

    @staticmethod
    def add_method_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("method-specific arguments")
        parser.add_argument(
            "-ssh",
            "--save-standard-histogram",
            default=None,
            type=save_file_path(),
            help="save the standard histogram fit by the method",
        )
        parser.add_argument(
            "-lsh",
            "--load-standard-histogram",
            default=None,
            type=file_path(),
            help="load a standard histogram previously fit by the method",
        )
        return parent_parser

    def call_from_argparse_args(self, args: Namespace) -> None:
        if args.load_standard_histogram is not None:
            self.load_standard_histogram(args.load_standard_histogram)
            self.fit = lambda *args, **kwargs: None  # type: ignore[assignment]
        super().call_from_argparse_args(args)
