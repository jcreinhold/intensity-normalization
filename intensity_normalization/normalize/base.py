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

from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    Namespace,
)
from typing import List, Optional, Tuple

import nibabel as nib

from intensity_normalization import VALID_MODALITIES
from intensity_normalization.parse import (
    CLI,
    dir_path,
    file_path,
    positive_float,
    save_nifti_path,
)
from intensity_normalization.type import (
    Array,
    ArrayOrNifti,
    NiftiImage,
    PathLike,
)
from intensity_normalization.util.io import gather_images_and_masks, glob_ext


class NormalizeBase(CLI):
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

    def normalize_from_filenames(
        self,
        image_path: PathLike,
        mask_path: Optional[PathLike] = None,
        out_path: Optional[PathLike] = None,
        modality: Optional[str] = None,
    ):
        image = nib.load(image_path)
        mask = nib.load(mask_path) if mask_path is not None else None
        if out_path is None:
            out_path = self.append_name_to_file(image_path)
        normalized = self.normalize_nifti(image, mask, modality)
        normalized.to_filename(out_path)

    def call_from_argparse_args(self, args: Namespace):
        self.normalize_from_filenames(
            args.image, args.mask, args.output, args.modality,
        )

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

    def _get_mask(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ):
        if mask is None:
            mask = self.skull_stripped_foreground(data)
        return mask > 0.0

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
            "image", type=file_path(), help="Path of image to normalize.",
        )
        parser.add_argument(
            "-m",
            "--mask",
            type=file_path(),
            default=None,
            help="Path of foreground mask for image.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=save_nifti_path(),
            default=None,
            help="Path to save normalized image.",
        )
        parser.add_argument(
            "-mo",
            "--modality",
            type=str,
            default=None,
            choices=VALID_MODALITIES,
            help="Modality of the image.",
        )
        parser.add_argument(
            "-n",
            "--norm-value",
            type=positive_float(),
            default=1.0,
            help="Reference value for normalization.",
        )
        options = parser.add_argument_group("Options")
        options.add_argument(
            "-p",
            "--plot-histogram",
            action="store_true",
            help="Plot the histogram of the normalized image.",
        )
        options.add_argument(
            "-v",
            "--verbosity",
            action="count",
            default=0,
            help="Increase output verbosity (e.g., -vv is more than -v).",
        )
        return parser

    @classmethod
    def from_argparse_args(cls, args: Namespace):
        return cls(args.norm_value)


class NormalizeSetBase(NormalizeBase):
    def fit(
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ):
        images, masks = self.before_fit(images, masks, modality, **kwargs)
        self._fit(images, masks, modality, **kwargs)

    def _fit(
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def before_fit(
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[Array], Optional[List[Array]]]:
        assert len(images) > 0
        if hasattr(images[0], "get_fdata"):
            images = [image.get_fdata() for image in images]
        if hasattr(masks[0], "get_fdata"):
            masks = [mask.get_fdata() for mask in masks]
        return images, masks

    def fit_from_directories(
        self,
        image_dir: PathLike,
        mask_dir: Optional[PathLike] = None,
        modality: Optional[str] = None,
        ext: str = "nii*",
        return_normalized: bool = False,
        **kwargs,
    ):
        images, masks = gather_images_and_masks(image_dir, mask_dir, ext)
        self.fit(images, masks, modality, **kwargs)
        if return_normalized:
            normalized = [
                self(image, mask, modality) for image, mask in zip(images, masks)
            ]
            return normalized

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        parser = ArgumentParser(
            description=desc, formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "image_dir",
            type=dir_path(),
            help="Path of directory of images to normalize.",
        )
        parser.add_argument(
            "-m",
            "--mask-dir",
            type=dir_path(),
            default=None,
            help="Path of directory of foreground masks corresponding to images.",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=dir_path(),
            default=None,
            help="Path of directory in which to save normalized images.",
        )
        parser.add_argument(
            "-mo",
            "--modality",
            type=str,
            default=None,
            choices=VALID_MODALITIES,
            help="Modality of the images.",
        )
        options = parser.add_argument_group("Options")
        options.add_argument(
            "-p",
            "--plot-histogram",
            action="store_true",
            help="Plot the histogram of the normalized image.",
        )
        options.add_argument(
            "-v",
            "--verbosity",
            action="count",
            default=0,
            help="Increase output verbosity (e.g., -vv is more than -v).",
        )
        return parser

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
