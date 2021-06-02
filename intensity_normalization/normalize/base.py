# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.base

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "NormalizeBase",
    "NormalizeSetBase",
]

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, Optional

import nibabel as nib

from intensity_normalization.type import (
    Array,
    ArrayOrNifti,
    NiftiImage,
    PathLike,
)
from intensity_normalization.util.io import gather_images_and_masks, split_filename


class NormalizeBase:
    def __init__(self, norm_value: float = 1.0):
        self.norm_value = norm_value

    def __call__(
        self,
        data: ArrayOrNifti,
        mask: Optional[ArrayOrNifti] = None,
        modality: Optional[str] = None,
    ) -> ArrayOrNifti:
        if isinstance(data, nib.Nifti1Image):
            return self.normalize_nifti(data, mask, modality)
        else:
            return self.normalize_array(data, mask, modality)

    def normalize_array(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> Array:
        self.setup(data, mask, modality)
        loc = self.calculate_location(data, mask, modality)
        scale = self.calculate_scale(data, mask, modality)
        self.teardown()
        return ((data - loc) / scale) * self.norm_value

    def normalize_nifti(
        self,
        image: NiftiImage,
        mask_image: Optional[NiftiImage] = None,
        modality: Optional[str] = None,
    ) -> NiftiImage:
        data = image.get_fdata()
        mask = mask_image and mask_image.get_fdata()
        normalized = self.normalize_array(data, mask, modality)
        return nib.Nifti1Image(normalized, image.affine, image.header)

    def calculate_location(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ) -> float:
        raise NotImplementedError

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ) -> float:
        raise NotImplementedError

    def setup(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ):
        return

    def teardown(self):
        return

    @staticmethod
    def estimate_foreground(data: Array) -> Array:
        return data > data.mean()

    @staticmethod
    def skull_stripped_foreground(data: Array) -> Array:
        return data > 0.0

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    def append_name_to_file(self, filepath: PathLike) -> Path:
        path, base, ext = split_filename(filepath)
        return path / (base + f"_{self.name()}" + ext)

    def _get_mask(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ):
        if mask is None:
            mask = self.skull_stripped_foreground(data)
        return mask

    def _get_voi(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ) -> Array:
        return data[self._get_mask(data, mask, modality)]

    def _get_modality(self, modality: Optional[str]) -> str:
        return "t1" if modality is None else modality.lower()

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        parser = ArgumentParser(
            description=desc, formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "image", type=str, help="path of image to normalize",
        )
        parser.add_argument(
            "--mask", type=str, default=None, help="path of foreground mask for image",
        )
        parser.add_argument(
            "--modality", type=str, default=None, help="modality of the MR image",
        )
        parser.add_argument(
            "--norm-value",
            type=float,
            default=1.0,
            help="reference value for normalization",
        )
        return parser

    @staticmethod
    def add_method_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        return parent_parser


class NormalizeSetBase(NormalizeBase):
    def fit(
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def fit_from_directories(
        self,
        image_dir: PathLike,
        mask_dir: Optional[PathLike] = None,
        modality: Optional[str] = None,
        ext: str = "nii*",
        **kwargs,
    ):
        images, masks = gather_images_and_masks(image_dir, mask_dir, ext)
        self.fit(images, masks, modality, **kwargs)

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        parser = ArgumentParser(
            description=desc, formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--image-dir",
            type=str,
            default=None,
            help="path of directory of images to normalize",
        )
        parser.add_argument(
            "--mask-dir",
            type=str,
            default=None,
            help="path of directory of foreground masks corresponding to images",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="path of directory in which to save normalized images",
        )
        parser.add_argument(
            "--modality", type=str, default=None, help="modality of the MR images",
        )
        return parser
