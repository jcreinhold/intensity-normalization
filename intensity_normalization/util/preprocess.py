"""Preprocess MR images for image processing

Preprocess MR images according to a simple scheme:
1) N4 bias field correction
2) resample to X mm x Y mm x Z mm
3) reorient images to specification

Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 21 May 2018
"""

from __future__ import annotations

__all__ = ["preprocess", "Preprocessor"]

import argparse
import builtins
import logging
import typing

import nibabel as nib

import intensity_normalization.base_cli as intnormcli
import intensity_normalization.typing as intnormt

logger = logging.getLogger(__name__)

try:
    import ants
except ImportError as ants_imp_exn:
    msg = "ANTsPy not installed. Install antspyx to use preprocessor."
    raise RuntimeError(msg) from ants_imp_exn


def preprocess(
    image: intnormt.Image,
    /,
    mask: intnormt.Image | None = None,
    *,
    resolution: typing.Tuple[builtins.float, ...] | None = None,
    orientation: builtins.str = "RAS",
    n4_convergence_options: typing.Dict[builtins.str, typing.Any] | None = None,
    interp_type: builtins.str = "linear",
    second_n4_with_smoothed_mask: builtins.bool = True,
) -> typing.Tuple[intnormt.Image, intnormt.Image]:
    """Preprocess an MR image

    Preprocess an MR image according to a simple scheme:
    1) N4 bias field correction
    2) resample to X mm x Y mm x ...
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
                interp_type=intnormt.interp_type_dict["nearest_neighbor"],
            )
        if resolution != image.spacing:
            logger.debug(f"Resampling image to {resolution}")
            image = ants.resample_image(
                image,
                resolution,
                use_voxels=False,
                interp_type=intnormt.interp_type_dict[interp_type],
            )
    image = image.reorient_image2(orientation)
    mask = mask.reorient_image2(orientation)
    image = image.to_nibabel()
    mask = mask.to_nibabel()
    return image, mask


class Preprocessor(intnormcli.CLI):
    def __init__(
        self,
        *,
        resolution: typing.Tuple[builtins.float, ...] | None = None,
        orientation: builtins.str = "RAI",
        n4_convergence_options: typing.Dict[builtins.str, typing.Any] | None = None,
        interp_type: builtins.str = "linear",
        second_n4_with_smoothed_mask: builtins.bool = True,
    ):
        self.resolution = resolution
        self.orientation = orientation
        self.n4_convergence_options = n4_convergence_options
        self.interp_type = interp_type
        self.second_n4_with_smoothed_mask = second_n4_with_smoothed_mask

    def __call__(
        self, image: intnormt.Image, /, mask: intnormt.Image | None = None, **kwargs
    ) -> intnormt.Image:
        preprocessed, _ = preprocess(
            image,
            mask,
            resolution=self.resolution,
            orientation=self.orientation,
            n4_convergence_options=self.n4_convergence_options,
            interp_type=self.interp_type,
            second_n4_with_smoothed_mask=self.second_n4_with_smoothed_mask,
        )
        return preprocessed

    @staticmethod
    def name() -> str:
        return "pp"

    @staticmethod
    def fullname() -> str:
        return "preprocess"

    @staticmethod
    def description() -> str:
        desc = "Basic preprocessing of an MR image: "
        desc += "bias field-correction, resampling, and reorientation."
        return desc

    @staticmethod
    def get_parent_parser(desc: builtins.str, **kwargs) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "image",
            type=intnormt.file_path(),
            help="Path of image to normalize.",
        )
        parser.add_argument(
            "-m",
            "--mask",
            type=intnormt.file_path(),
            default=None,
            help="Path of foreground mask for image.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=intnormt.save_nifti_path(),
            default=None,
            help="Path to save registered image.",
        )
        parser.add_argument(
            "-r",
            "--resolution",
            nargs="+",
            type=intnormt.positive_float(),
            default=None,
            help="Resolution to resample image in mm per dimension",
        )
        parser.add_argument(
            "-or",
            "--orientation",
            type=str,
            choices=intnormt.allowed_orientations,
            default="RAI",
            help="Reorient image to this specification.",
            metavar="",
        )
        parser.add_argument(
            "-it",
            "--interp-type",
            type=str,
            choices=set(intnormt.interp_type_dict.keys()),
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
        parser.add_argument(
            "--version",
            action="store_true",
            help="print the version of intensity-normalization",
        )
        return parser

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace) -> Preprocessor:
        return cls(
            resolution=args.resolution,
            orientation=args.orientation,
            n4_convergence_options=None,
            interp_type=args.interp_type,
            second_n4_with_smoothed_mask=args.second_n4_with_smoothed_mask,
        )

    @staticmethod
    def load_image(image_path: intnormt.PathLike) -> ants.ANTsImage:
        return ants.image_read(image_path)
