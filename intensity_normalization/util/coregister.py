"""Co-register images with ANTsPy
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 03 Jun 2021
"""

from __future__ import annotations

__all__ = ["register", "Registrator"]

import argparse
import collections.abc
import logging
import typing

import nibabel as nib
import numpy as np

import intensity_normalization as intnorm
import intensity_normalization.base_cli as intnormcli
import intensity_normalization.typing as intnormt

logger = logging.getLogger(__name__)

try:
    import ants
except ImportError as ants_imp_exn:
    msg = "ANTsPy not installed. Install antspyx to use co-registration."
    raise RuntimeError(msg) from ants_imp_exn

ValidImage = typing.Union[nib.nifti1.Nifti1Image, ants.ANTsImage, intnormt.ImageLike]


def to_ants(image: ValidImage, /) -> ants.ANTsImage:
    if isinstance(image, ants.ANTsImage):
        ants_image = image
    elif isinstance(image, nib.nifti1.Nifti1Image):
        ants_image = ants.from_nibabel(image)
    elif isinstance(image, np.ndarray):
        ants_image = ants.from_numpy(image)
    else:
        msg = "Provided image must be an ANTsImage, Nifti1Image,"
        msg += f" or (a subclass of) np.ndarray. Got '{type(image)}'."
        raise ValueError(msg)
    return ants_image


def register(
    image: ValidImage,
    /,
    template: typing.Optional[ValidImage] = None,
    *,
    type_of_transform: str = "Affine",
    interpolator: str = "bSpline",
    metric: str = "mattes",
    initial_rigid: bool = True,
    template_mask: typing.Optional[ValidImage] = None,
) -> nib.nifti1.Nifti1Image | ants.ANTsImage:
    if template is None:
        standard_mni = ants.get_ants_data("mni")
        template = ants.image_read(standard_mni)
    else:
        template = to_ants(template)
    is_nibabel = isinstance(image, nib.nifti1.Nifti1Image)
    image = to_ants(image)
    if initial_rigid:
        logger.debug("Doing initial rigid registration.")
        transforms = ants.registration(
            fixed=template,
            moving=image,
            type_of_transform="Rigid",
            aff_metric=metric,
            syn_metric=metric,
        )
        rigid_transform = transforms["fwdtransforms"][0]
    else:
        rigid_transform = None
    logger.debug(f"Doing {type_of_transform} registration.")
    transform = ants.registration(
        fixed=template,
        moving=image,
        initial_transform=rigid_transform,
        type_of_transform=type_of_transform,
        mask=template_mask,
        aff_metric=metric,
        syn_metric=metric,
    )["fwdtransforms"]
    logger.debug("Applying transformations.")
    registered = ants.apply_transforms(
        template,
        image,
        transform,
        interpolator=interpolator,
    )
    return registered.to_nibabel() if is_nibabel else registered


class Registrator(intnormcli.SingleImageCLI):
    def __init__(
        self,
        template: nib.nifti1.Nifti1Image | ants.ANTsImage = None,
        *,
        type_of_transform: str = "Affine",
        interpolator: str = "bSpline",
        metric: str = "mattes",
        initial_rigid: bool = True,
    ):
        super().__init__()
        if template is None:
            logger.info("Using MNI (in RAS orientation) as template.")
            standard_mni = ants.get_ants_data("mni")
            self.template = ants.image_read(standard_mni).reorient_image2("RAS")
        else:
            logger.debug("Loading template.")
            self.template = ants.from_nibabel(template)
        self.type_of_transform = type_of_transform
        self.interpolator = interpolator
        self.metric = metric
        self.initial_rigid = initial_rigid

    def __call__(
        self,
        image: nib.nifti1.Nifti1Image | ants.ANTsImage,
        /,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> nib.nifti1.Nifti1Image | ants.ANTsImage:
        return register(
            image,
            template=self.template,
            type_of_transform=self.type_of_transform,
            interpolator=self.interpolator,
            metric=self.metric,
            initial_rigid=self.initial_rigid,
        )

    def register_images(
        self,
        images: collections.abc.Sequence[nib.nifti1.Nifti1Image | ants.ANTsImage],
        /,
    ) -> collections.abc.Sequence[nib.nifti1.Nifti1Image | ants.ANTsImage]:
        return [self(image) for image in images]

    def register_images_to_templates(
        self,
        images: collections.abc.Sequence[nib.nifti1.Nifti1Image | ants.ANTsImage],
        /,
        *,
        templates: collections.abc.Sequence[nib.nifti1.Nifti1Image | ants.ANTsImage],
    ) -> collections.abc.Sequence[nib.nifti1.Nifti1Image | ants.ANTsImage]:
        assert len(images) == len(templates)
        registered = []
        original_template = self.template
        for image, template in zip(images, templates):
            self.template = template
            registered.append(self(image))
        self.template = original_template
        return registered

    @staticmethod
    def name() -> str:
        return "registered"

    @staticmethod
    def fullname() -> str:
        return Registrator.name()

    @staticmethod
    def description() -> str:
        return "Co-register an image to MNI or another image."

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
            "-t",
            "--template",
            type=intnormt.file_path(),
            default=None,
            help="Path of target for registration.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=intnormt.save_file_path(),
            default=None,
            help="Path to save registered image.",
        )
        parser.add_argument(
            "-tot",
            "--type-of-transform",
            type=str,
            default="Affine",
            choices=intnormt.allowed_transforms,
            help="Type of registration transform to perform.",
            metavar="",  # avoid printing massive list of choices
        )
        parser.add_argument(
            "-i",
            "--interpolator",
            type=str,
            default="bSpline",
            choices=intnormt.allowed_interpolators,
            help="Type of interpolator to use.",
            metavar="",
        )
        parser.add_argument(
            "-mc",
            "--metric",
            type=str,
            default="mattes",
            choices=intnormt.allowed_metrics,
            help="Metric to use for registration loss function.",
            metavar="",
        )
        parser.add_argument(
            "-ir",
            "--initial-rigid",
            action="store_true",
            help="Do a rigid registration before doing "
            "the `type_of_transform` registration.",
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
    def from_argparse_args(cls, args: argparse.Namespace) -> Registrator:
        if args.template is not None:
            args.template = ants.image_read(args.template)
        return cls(
            template=args.template,
            type_of_transform=args.type_of_transform,
            interpolator=args.interpolator,
            metric=args.metric,
            initial_rigid=args.initial_rigid,
        )

    @staticmethod
    def load_image(image_path: intnormt.PathLike) -> ants.ANTsImage:
        return ants.image_read(image_path)
