# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.base

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "NormalizeBase",
    "NormalizeFitBase",
]

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import List, Optional, Tuple, Type, TypeVar

import nibabel as nib

from intensity_normalization import VALID_MODALITIES
from intensity_normalization.parse import CLIParser
from intensity_normalization.plot.histogram import HistogramPlotter, plot_histogram
from intensity_normalization.type import (
    Array,
    ArrayOrNifti,
    NiftiImage,
    PathLike,
    dir_path,
    file_path,
    positive_float,
    save_nifti_path,
)
from intensity_normalization.util.io import gather_images_and_masks, glob_ext

NB = TypeVar("NB", bound="NormalizeBase")

logger = logging.getLogger(__name__)


class NormalizeBase(CLIParser):
    def __init__(self, norm_value: float = 1.0):
        self.norm_value = norm_value

    def __call__(  # type: ignore[override]
        self,
        data: ArrayOrNifti,
        mask: Optional[ArrayOrNifti] = None,
        modality: Optional[str] = None,
    ) -> ArrayOrNifti:
        if isinstance(data, NiftiImage):
            return self.normalize_nifti(data, mask, modality)
        else:
            return self.normalize_array(data, mask, modality)

    def normalize_array(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> Array:
        self.setup(data, mask, modality)
        loc = self.calculate_location(data, mask, modality)
        scale = self.calculate_scale(data, mask, modality)
        self.teardown()
        normalized: Array = ((data - loc) / scale) * self.norm_value
        return normalized

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
    ) -> NiftiImage:
        image = nib.load(image_path)
        mask = nib.load(mask_path) if mask_path is not None else None
        if out_path is None:
            out_path = self.append_name_to_file(image_path)
        logger.info(f"Normalizing image: {image_path}")
        normalized = self.normalize_nifti(image, mask, modality)
        logger.info(f"Saving normalized image: {out_path}")
        normalized.to_filename(out_path)
        return normalized, mask

    def plot_histogram(
        self,
        args: Namespace,
        normalized: NiftiImage,
        mask: Optional[NiftiImage] = None,
    ) -> None:
        from pathlib import Path

        import matplotlib.pyplot as plt

        if args.output is None:
            output = Path(args.image).parent / "hist.pdf"
        else:
            output = Path(args.output).parent / "hist.pdf"
        normalized_data = normalized.get_fdata()
        mask_data = mask and mask.get_fdata()
        ax = plot_histogram(normalized_data, mask_data)
        ax.set_title(self.fullname())
        plt.savefig(output)

    def call_from_argparse_args(self, args: Namespace) -> None:
        normalized, mask = self.normalize_from_filenames(
            args.image,
            args.mask,
            args.output,
            args.modality,
        )
        if args.plot_histogram:
            self.plot_histogram(args, normalized, mask)
        self.save_additional_info(args, normalized=normalized, mask=mask)

    def calculate_location(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> float:
        raise NotImplementedError

    def calculate_scale(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> float:
        raise NotImplementedError

    def setup(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> None:
        return

    def teardown(self) -> None:
        return

    @staticmethod
    def estimate_foreground(data: Array) -> Array:
        foreground: Array = data > data.mean()
        return foreground

    @staticmethod
    def skull_stripped_foreground(data: Array) -> Array:
        if data.min() < 0.0:
            msg = (
                "Data contains negative values; "
                "skull-stripped functionality assumes "
                "the foreground is all positive. "
                "Provide the brain mask if otherwise."
            )
            logger.warning(msg)
        ss_foreground: Array = data > 0.0
        return ss_foreground

    def _get_mask(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> Array:
        if mask is None:
            mask = self.skull_stripped_foreground(data)
        out: Array = mask > 0.0
        return out

    def _get_voi(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> Array:
        voi: Array = data[self._get_mask(data, mask, modality)]
        return voi

    @staticmethod
    def _get_modality(modality: Optional[str]) -> str:
        return "t1" if modality is None else modality.lower()

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        parser = ArgumentParser(
            description=desc,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "image",
            type=file_path(),
            help="Path of image to normalize.",
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
        parser.add_argument(
            "-p",
            "--plot-histogram",
            action="store_true",
            help="Plot the histogram of the normalized image.",
        )
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            default=0,
            help="Increase output verbosity (e.g., -vv is more than -v).",
        )
        return parser

    @classmethod
    def from_argparse_args(cls: Type[NB], args: Namespace) -> NB:
        return cls(args.norm_value)

    def save_additional_info(  # type: ignore[no-untyped-def]
        self,
        args: Namespace,
        **kwargs,
    ) -> None:
        return


NSB = TypeVar("NSB", bound="NormalizeSampleBase")


class NormalizeSampleBase(NormalizeBase):
    def fit(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    def process_directories(  # type: ignore[no-untyped-def]
        self,
        image_dir: PathLike,
        mask_dir: Optional[PathLike] = None,
        modality: Optional[str] = None,
        ext: str = "nii*",
        return_normalized_and_masks: bool = False,
        **kwargs,
    ) -> Optional[Tuple[List[ArrayOrNifti], List[Optional[ArrayOrNifti]]]]:
        logger.debug("Grabbing images")
        images, masks = gather_images_and_masks(image_dir, mask_dir, ext)
        self.fit(images, masks, modality, **kwargs)
        if return_normalized_and_masks:
            normalized: List[ArrayOrNifti] = []
            n_images = len(images)
            assert n_images == len(masks)
            for i, (image, mask) in enumerate(zip(images, masks), 1):
                logger.info(f"Normalizing image {i}/{n_images}")
                normalized.append(self(image, mask, modality))
            return normalized, masks
        return None

    def plot_histograms(
        self,
        args: Namespace,
        normalized: List[NiftiImage],
        masks: List[Optional[NiftiImage]],
    ) -> None:
        from pathlib import Path

        import matplotlib.pyplot as plt

        if args.output_dir is None:
            output = Path(args.image_dir) / "hist.pdf"
        else:
            output = Path(args.output_dir) / "hist.pdf"
        hp = HistogramPlotter(title=self.fullname())
        _ = hp(normalized, masks)
        plt.savefig(output)

    def call_from_argparse_args(self, args: Namespace) -> None:
        normalized, masks = self.process_directories(  # type: ignore[misc]
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            modality=args.modality,
            ext=args.extension,
            return_normalized_and_masks=True,
        )
        assert isinstance(normalized, list)
        image_filenames = glob_ext(args.image_dir)
        output_filenames = [
            self.append_name_to_file(fn, args.output_dir) for fn in image_filenames
        ]
        n_images = len(normalized)
        assert n_images == len(output_filenames)
        for i, (norm_image, fn) in enumerate(zip(normalized, output_filenames), 1):
            assert isinstance(norm_image, NiftiImage)
            logger.info(f"Saving normalized image: {fn} ({i}/{n_images})")
            norm_image.to_filename(fn)
        self.save_additional_info(
            args,
            normalized=normalized,
            masks=masks,
            image_filenames=image_filenames,
        )
        if args.plot_histogram:
            self.plot_histograms(args, normalized, masks)

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        parser = ArgumentParser(
            description=desc,
            formatter_class=ArgumentDefaultsHelpFormatter,
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
        parser.add_argument(
            "-e",
            "--extension",
            type=str,
            default="nii*",
            help="Extension of images (must be nibabel readable).",
        )
        parser.add_argument(
            "-p",
            "--plot-histogram",
            action="store_true",
            help="Plot the histogram of the normalized image.",
        )
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            default=0,
            help="Increase output verbosity (e.g., -vv is more than -v).",
        )
        return parser

    @classmethod
    def from_argparse_args(cls: Type[NSB], args: Namespace) -> NSB:
        out: NSB = cls()
        return out


class NormalizeFitBase(NormalizeSampleBase):
    def fit(  # type: ignore[no-untyped-def,override]
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> None:
        images, masks = self.before_fit(images, masks, modality, **kwargs)
        logger.info("Fitting")
        self._fit(images, masks, modality, **kwargs)
        logger.debug("Done fitting")

    def _fit(  # type: ignore[no-untyped-def]
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def before_fit(  # type: ignore[no-untyped-def]
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[Array], Optional[List[Array]]]:
        assert len(images) > 0
        logger.info("Loading data")
        if hasattr(images[0], "get_fdata"):
            images = [img.get_fdata() for img in images]
        if masks is not None:
            if hasattr(masks[0], "get_fdata"):
                masks = [msk.get_fdata() for msk in masks]
        logger.debug("Loaded data")
        return images, masks

    def fit_from_directories(  # type: ignore[no-untyped-def]
        self,
        image_dir: PathLike,
        mask_dir: Optional[PathLike] = None,
        modality: Optional[str] = None,
        ext: str = "nii*",
        return_normalized_and_masks: bool = False,
        **kwargs,
    ) -> Optional[Tuple[List[ArrayOrNifti], List[Optional[ArrayOrNifti]]]]:
        return self.process_directories(
            image_dir=image_dir,
            mask_dir=mask_dir,
            modality=modality,
            ext=ext,
            return_normalized_and_masks=return_normalized_and_masks,
            **kwargs,
        )
