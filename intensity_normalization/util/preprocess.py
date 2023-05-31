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
import logging
import typing

import nibabel as nib
import numpy as np
import pymedio.image as mioi

import intensity_normalization as intnorm
import intensity_normalization.base_cli as intnormcli
import intensity_normalization.typing as intnormt

logger = logging.getLogger(__name__)

try:
    import ants
except ImportError as ants_imp_exn:
    msg = "ANTsPy not installed. Install antspyx to use preprocessor."
    raise RuntimeError(msg) from ants_imp_exn


def preprocess(
    image: intnormt.ImageLike,
    /,
    mask: intnormt.ImageLike | None = None,
    *,
    resolution: tuple[float, ...] | None = None,
    orientation: str = "RAS",
    n4_convergence_options: dict[str, typing.Any] | None = None,
    interp_type: str = "linear",
    second_n4_with_smoothed_mask: bool = True,
) -> tuple[mioi.Image, mioi.Image]:
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
    logger.debug(f"N4 Options are: {n4_convergence_options}.")

    ants_image = _to_ants(image)

    if mask is not None:
        ants_mask = _to_ants(mask)
    else:
        ants_mask = ants_image.get_mask()

    logger.debug("Starting bias field correction.")
    ants_image = ants.n4_bias_field_correction(
        ants_image, convergence=n4_convergence_options
    )
    if second_n4_with_smoothed_mask:
        smoothed_mask = ants.smooth_image(ants_mask, 1.0)
        logger.debug("Starting 2nd bias field correction.")
        ants_image = ants.n4_bias_field_correction(
            ants_image,
            convergence=n4_convergence_options,
            weight_mask=smoothed_mask,
        )
    if resolution is not None:
        if resolution != ants_mask.spacing:
            logger.debug(f"Resampling mask to {resolution}.")
            ants_mask = ants.resample_image(
                ants_mask,
                resolution,
                use_voxels=False,
                interp_type=intnormt.interp_type_dict["nearest_neighbor"],
            )
        if resolution != ants_image.spacing:
            logger.debug(f"Resampling image to {resolution}")
            ants_image = ants.resample_image(
                ants_image,
                resolution,
                use_voxels=False,
                interp_type=intnormt.interp_type_dict[interp_type],
            )
    ants_image = ants_image.reorient_image2(orientation)
    ants_mask = ants_mask.reorient_image2(orientation)
    _image = ants_image.to_nibabel()
    pp_image: mioi.Image = mioi.Image(_image.get_fdata(), _image.affine)
    _mask = ants_mask.to_nibabel()
    pp_mask: mioi.Image = mioi.Image(_mask.get_fdata(), _mask.affine)
    return pp_image, pp_mask


class Preprocessor(intnormcli.SingleImageCLI):
    def __init__(
        self,
        *,
        resolution: tuple[float, ...] | None = None,
        orientation: str = "RAI",
        n4_convergence_options: dict[str, typing.Any] | None = None,
        interp_type: str = "linear",
        second_n4_with_smoothed_mask: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.orientation = orientation
        self.n4_convergence_options = n4_convergence_options
        self.interp_type = interp_type
        self.second_n4_with_smoothed_mask = second_n4_with_smoothed_mask

    def __call__(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        **kwargs: typing.Any,
    ) -> intnormt.ImageLike:
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

    @classmethod
    def get_parent_parser(
        cls,
        desc: str,
        valid_modalities: frozenset[str] = intnorm.VALID_MODALITIES,
        **kwargs: typing.Any,
    ) -> argparse.ArgumentParser:
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
            type=intnormt.save_file_path(),
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
            help="Print the version of intensity-normalization.",
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


def _to_ants(image: typing.Any) -> ants.ANTsImage:
    if isinstance(image, nib.nifti1.Nifti1Image):
        ants_image = ants.from_nibabel(image)
    elif isinstance(image, mioi.Image):
        ants_image = ants.from_numpy(
            image, origin=image.origin, spacing=image.spacing, direction=image.direction
        )
    elif isinstance(image, np.ndarray):
        ants_image = ants.from_numpy(image)
    elif isinstance(image, ants.ANTsImage):
        ants_image = image
    else:
        try:
            ants_image = ants.ANTsImage(image.numpy())
        except Exception:
            raise ValueError("Unexpected image type.")
    return ants_image
