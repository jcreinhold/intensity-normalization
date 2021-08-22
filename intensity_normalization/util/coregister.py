# -*- coding: utf-8 -*-
"""
intensity-normalization.util.coregister

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 03, 2021
"""

__all__ = [
    "register",
    "Registrator",
]

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import List, Optional, Type, TypeVar, Union

from intensity_normalization.parse import CLIParser
from intensity_normalization.type import (
    Array,
    ArrayOrNifti,
    NiftiImage,
    PathLike,
    allowed_interpolators,
    allowed_metrics,
    allowed_transforms,
    file_path,
    save_nifti_path,
)

logger = logging.getLogger(__name__)

try:
    import ants
except (ModuleNotFoundError, ImportError):
    logger.error("ANTsPy not installed. Install antspyx to use co-registration.")
    raise


def to_ants(image: Union[ArrayOrNifti, ants.ANTsImage]) -> ants.ANTsImage:
    if isinstance(image, ants.ANTsImage):
        ants_image = image
    elif isinstance(image, NiftiImage):
        ants_image = ants.from_nibabel(image)
    elif isinstance(image, Array):
        ants_image = ants.from_numpy(image)
    else:
        raise ValueError(
            "Provided image must be an ANTsImage, Nifti1Image, or np.ndarray."
            f" Got {type(image)}."
        )
    return ants_image


def register(
    image: Union[NiftiImage, ants.ANTsImage],
    template: Optional[Union[NiftiImage, ants.ANTsImage]] = None,
    type_of_transform: str = "Affine",
    interpolator: str = "bSpline",
    metric: str = "mattes",
    initial_rigid: bool = True,
    template_mask: Optional[Union[NiftiImage, ants.ANTsImage]] = None,
) -> Union[NiftiImage, ants.ANTsImage]:
    if template is None:
        standard_mni = ants.get_ants_data("mni")
        template = ants.image_read(standard_mni)
    else:
        template = to_ants(template)
    is_nibabel = isinstance(image, NiftiImage)
    image = to_ants(image)
    if initial_rigid:
        logger.debug("Doing initial rigid registration")
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
    logger.debug(f"Doing {type_of_transform} registration")
    transform = ants.registration(
        fixed=template,
        moving=image,
        initial_transform=rigid_transform,
        type_of_transform=type_of_transform,
        mask=template_mask,
        aff_metric=metric,
        syn_metric=metric,
    )["fwdtransforms"]
    logger.debug("Applying transformations")
    registered = ants.apply_transforms(
        template,
        image,
        transform,
        interpolator=interpolator,
    )
    return registered.to_nibabel() if is_nibabel else registered


R = TypeVar("R", bound="Registrator")


class Registrator(CLIParser):
    def __init__(
        self,
        template: Optional[NiftiImage] = None,
        type_of_transform: str = "Affine",
        interpolator: str = "bSpline",
        metric: str = "mattes",
        initial_rigid: bool = True,
    ):
        if template is None:
            logger.info("Using MNI (in RAS orientation) as template")
            standard_mni = ants.get_ants_data("mni")
            self.template = ants.image_read(standard_mni).reorient_image2("RAS")
        else:
            logger.debug("Loading template")
            self.template = ants.from_nibabel(template)
        self.type_of_transform = type_of_transform
        self.interpolator = interpolator
        self.metric = metric
        self.initial_rigid = initial_rigid

    def __call__(  # type: ignore[no-untyped-def,override]
        self,
        image: NiftiImage,
        *args,
        **kwargs,
    ) -> NiftiImage:
        return register(
            image=image,
            template=self.template,
            type_of_transform=self.type_of_transform,
            interpolator=self.interpolator,
            metric=self.metric,
            initial_rigid=self.initial_rigid,
        )

    def register_images(self, images: List[NiftiImage]) -> List[NiftiImage]:
        return [self(image) for image in images]

    def register_images_to_templates(
        self,
        images: List[NiftiImage],
        templates: List[NiftiImage],
    ) -> List[NiftiImage]:
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
    def description() -> str:
        return "Co-register an image to MNI or another image."

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
            "-t",
            "--template",
            type=file_path(),
            default=None,
            help="Path of target for registration.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=save_nifti_path(),
            default=None,
            help="Path to save registered image.",
        )
        parser.add_argument(
            "-tot",
            "--type-of-transform",
            type=str,
            default="Affine",
            choices=allowed_transforms,
            help="Type of registration transform to perform.",
            metavar="",
        )
        parser.add_argument(
            "-i",
            "--interpolator",
            type=str,
            default="bSpline",
            choices=allowed_interpolators,
            help="Type of interpolator to use.",
            metavar="",
        )
        parser.add_argument(
            "-mc",
            "--metric",
            type=str,
            default="mattes",
            choices=allowed_metrics,
            help="Metric to use for registration loss function.",
            metavar="",
        )
        parser.add_argument(
            "-ir",
            "--initial-rigid",
            action="store_true",
            help=(
                "Do a rigid registration before doing "
                "the `type_of_transform` registration."
            ),
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
    def from_argparse_args(cls: Type[R], args: Namespace) -> R:
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
    def load_image(image_path: PathLike) -> ants.ANTsImage:
        return ants.image_read(image_path)
