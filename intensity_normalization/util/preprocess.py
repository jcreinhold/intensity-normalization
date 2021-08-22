# -*- coding: utf-8 -*-
"""
intensity_normalization.util.preprocess

Preprocess MR images according to a simple scheme:
1) N4 bias field correction
2) resample to X mm x Y mm x Z mm
3) reorient images to specification

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 21, 2018
"""

__all__ = [
    "preprocess",
    "Preprocessor",
]

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Optional, Tuple, Type, TypeVar

import nibabel as nib

from intensity_normalization.parse import CLIParser
from intensity_normalization.type import (
    NiftiImage,
    PathLike,
    allowed_orientations,
    file_path,
    interp_type_dict,
    positive_float,
    save_nifti_path,
)

logger = logging.getLogger(__name__)

try:
    import ants
except (ModuleNotFoundError, ImportError):
    logger.error("ANTsPy not installed. Install antspyx to use preprocessor.")
    raise


def preprocess(
    image: NiftiImage,
    mask: Optional[NiftiImage] = None,
    resolution: Optional[Tuple[float, float, float]] = None,
    orientation: str = "RAS",
    n4_convergence_options: Optional[dict] = None,
    interp_type: str = "linear",
    second_n4_with_smoothed_mask: bool = True,
) -> Tuple[NiftiImage, NiftiImage]:
    """Preprocess an MR image

    Preprocess an MR image according to a simple scheme:
    1) N4 bias field correction
    2) resample to X mm x Y mm x Z mm
    3) reorient images to RAI

    Args:
        image: image to preprocess
        mask: mask covering the brain of image (none if already skull-stripped)
        resolution: resolution for resampling. None for no resampling.
        orientation: reorient the image according to this. See ANTsPy for details.
        n4_convergence_options: n4 processing options. See ANTsPy for details.
        interp_type: interpolation type for resampling
            choices: linear, nearest_neighbor, gaussian, windowed_sinc, bspline
        second_n4_with_smoothed_mask: do a second N4 with a smoothed mask
            often improves the bias field correction in the image

    Returns:
        preprocessed image and corresponding foreground mask
    """

    if n4_convergence_options is None:
        n4_convergence_options = {"iters": [200, 200, 200, 200], "tol": 1e-7}
    logger.debug(f"N4 Options are: {n4_convergence_options}")

    if isinstance(image, nib.Nifti1Image):
        image = ants.from_nibabel(image)
    if mask is not None:
        if isinstance(mask, nib.Nifti1Image):
            mask = ants.from_nibabel(mask)
    else:
        mask = image.get_mask()
    logger.debug("Starting bias field correction")
    image = ants.n4_bias_field_correction(image, convergence=n4_convergence_options)
    if second_n4_with_smoothed_mask:
        smoothed_mask = ants.smooth_image(mask, 1.0)
        logger.debug("Starting 2nd bias field correction")
        image = ants.n4_bias_field_correction(
            image,
            convergence=n4_convergence_options,
            weight_mask=smoothed_mask,
        )
    if resolution is not None:
        if resolution != mask.spacing:
            logger.debug(f"Resampling mask to {resolution}")
            mask = ants.resample_image(
                mask,
                resolution,
                use_voxels=False,
                interp_type=interp_type_dict["nearest_neighbor"],
            )
        if resolution != image.spacing:
            logger.debug(f"Resampling image to {resolution}")
            image = ants.resample_image(
                image,
                resolution,
                use_voxels=False,
                interp_type=interp_type_dict[interp_type],
            )
    image = image.reorient_image2(orientation)
    mask = mask.reorient_image2(orientation)
    image = image.to_nibabel()
    mask = mask.to_nibabel()
    return image, mask


PP = TypeVar("PP", bound="Preprocessor")


class Preprocessor(CLIParser):
    def __init__(
        self,
        resolution: Optional[Tuple[float, float, float]] = None,
        orientation: str = "RAI",
        n4_convergence_options: Optional[dict] = None,
        interp_type: str = "linear",
        second_n4_with_smoothed_mask: bool = True,
    ):
        self.resolution = resolution
        self.orientation = orientation
        self.n4_convergence_options = n4_convergence_options
        self.interp_type = interp_type
        self.second_n4_with_smoothed_mask = second_n4_with_smoothed_mask

    def __call__(  # type: ignore[override]
        self,
        image: NiftiImage,
        mask: Optional[NiftiImage] = None,
    ) -> NiftiImage:
        preprocessed, _ = preprocess(
            image,
            mask,
            self.resolution,
            self.orientation,
            self.n4_convergence_options,
            self.interp_type,
            self.second_n4_with_smoothed_mask,
        )
        return preprocessed

    @staticmethod
    def name() -> str:
        return "pp"

    @staticmethod
    def description() -> str:
        return (
            "Basic preprocessing of an MR image: "
            "bias field-correction, resampling, and reorientation."
        )

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
            help="Path to save registered image.",
        )
        parser.add_argument(
            "-r",
            "--resolution",
            nargs="+",
            type=positive_float(),
            default=None,
            help="Resolution to resample image in mm per dimension",
        )
        parser.add_argument(
            "-or",
            "--orientation",
            type=str,
            choices=allowed_orientations,
            default="RAI",
            help="Reorient image to this specification.",
            metavar="",
        )
        parser.add_argument(
            "-it",
            "--interp-type",
            type=str,
            choices=set(interp_type_dict.keys()),
            default="linear",
            help="Use this interpolator for resampling.",
            metavar="",
        )
        parser.add_argument(
            "-2n4",
            "--second-n4-with-smoothed-mask",
            action="store_true",
            help="Do a second N4 bias field-correction with a smoothed mask.",
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
    def from_argparse_args(cls: Type[PP], args: Namespace) -> PP:
        return cls(
            args.resolution,
            args.orientation,
            None,
            args.interp_type,
            args.second_n4_with_smoothed_mask,
        )

    @staticmethod
    def load_image(image_path: PathLike) -> ants.ANTsImage:
        return ants.image_read(image_path)
