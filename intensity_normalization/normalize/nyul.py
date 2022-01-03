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
from typing import List, Optional, Type, TypeVar

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

NN = TypeVar("NN", bound="NyulNormalize")


class NyulNormalize(NormalizeFitBase):
    """
    Args:
        output_min_value: where min-percentile mapped for output normalized image
        output_max_value: where max-percentile mapped for output normalized image
        min_percentile: min percentile to account for while finding
            standard histogram
        max_percentile: max percentile to account for while finding
            standard histogram
        next_percentile_after_min: next percentile after min for finding
            standard histogram (percentile-step creates intermediate percentiles)
        prev_percentile_before_max: previous percentile before max for finding
            standard histogram (percentile-step creates intermediate percentiles)
        percentile_step: percentile steps btwn next-percentile-after-min and
             prev-percentile-before-max for finding standard histogram
    """

    def __init__(
        self,
        *,
        output_min_value: float = 1.0,
        output_max_value: float = 100.0,
        min_percentile: float = 1.0,
        max_percentile: float = 99.0,
        percentile_after_min: float = 10.0,
        percentile_before_max: float = 90.0,
        percentile_step: float = 10.0,
    ):
        super().__init__()
        self.output_min_value = output_min_value
        self.output_max_value = output_max_value
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.percentile_after_min = percentile_after_min
        self.percentile_before_max = percentile_before_max
        self.percentile_step = percentile_step
        self._percentiles: Optional[Vector] = None

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
                self.percentile_after_min,
                self.percentile_before_max + self.percentile_step,
                self.percentile_step,
            )
            self._percentiles = np.concatenate(
                ([self.min_percentile], percs, [self.max_percentile])  # type: ignore[arg-type]
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
            min_p = np.percentile(voi, self.min_percentile)
            max_p = np.percentile(voi, self.max_percentile)
            f = interp1d([min_p, max_p], [self.output_min_value, self.output_max_value])
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
        parser.add_argument(
            "--output-min-value",
            type=float,
            default=1.0,
            help="where min-percentile mapped for output normalized image",
        )
        parser.add_argument(
            "--output-max-value",
            type=float,
            default=100.0,
            help="where max-percentile mapped for output normalized image",
        )
        parser.add_argument(
            "--min-percentile",
            type=float,
            default=1.0,
            help="min percentile to account for while finding standard histogram",
        )
        parser.add_argument(
            "--max-percentile",
            type=float,
            default=99.0,
            help="max percentile to account for while finding standard histogram",
        )
        parser.add_argument(
            "--percentile-after-min",
            type=float,
            default=10.0,
            help="percentile after min for finding standard histogram "
            "(percentile-step creates intermediate percentiles between "
            "this and percentile-before-max)",
        )
        parser.add_argument(
            "--percentile-before-max",
            type=float,
            default=90.0,
            help="percentile before max for finding standard histogram "
            "(percentile-step creates intermediate percentiles between "
            "this and percentile-after-min)",
        )
        parser.add_argument(
            "--percentile-step",
            type=float,
            default=10.0,
            help="percentile steps btwn next-percentile-after-min and "
            "prev-percentile-before-max for finding standard histogram",
        )
        return parent_parser

    def call_from_argparse_args(self, args: Namespace) -> None:
        if args.load_standard_histogram is not None:
            self.load_standard_histogram(args.load_standard_histogram)
            self.fit = lambda *args, **kwargs: None  # type: ignore[assignment]
        super().call_from_argparse_args(args)

    @classmethod
    def from_argparse_args(cls: Type[NN], args: Namespace) -> NN:
        out: NN = cls(
            output_min_value=args.output_min_value,
            output_max_value=args.output_max_value,
            min_percentile=args.min_percentile,
            max_percentile=args.max_percentile,
            percentile_after_min=args.percentile_after_min,
            percentile_before_max=args.percentile_before_max,
            percentile_step=args.percentile_step,
        )
        return out
