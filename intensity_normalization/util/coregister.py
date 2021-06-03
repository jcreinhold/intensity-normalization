# -*- coding: utf-8 -*-
"""
intensity-normalization.util.coregister

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 03, 2021
"""

__all__ = [
    "Registrator",
]

import logging
from typing import List, Optional


from intensity_normalization.type import NiftiImage

logger = logging.getLogger(__name__)

try:
    import ants
except (ModuleNotFoundError, ImportError):
    logger.warning("ANTsPy not installed. Install antspyx to use co-registration.")
    raise


class Registrator:
    def __init__(
        self,
        template: Optional[NiftiImage] = None,
        type_of_transform: str = "Affine",
        interpolator: str = "bSpline",
        initial_rigid: bool = True,
    ):
        if template is None:
            standard_mni = ants.get_ants_data("mni")
            self.template = ants.image_read(standard_mni)
        else:
            self.template = ants.from_nibabel(template)
        self.type_of_transform = type_of_transform
        self.interpolator = interpolator
        self.initial_rigid = initial_rigid

    def __call__(self, image: NiftiImage) -> NiftiImage:
        return self.register(image)

    def register(self, image: NiftiImage) -> NiftiImage:
        image = ants.from_nibabel(image)
        if self.initial_rigid:
            transforms = ants.registration(
                fixed=self.template, moving=image, type_of_transform="Rigid",
            )
            rigid_transform = transforms["fwdtransforms"][0]
        else:
            rigid_transform = None
        transform = ants.registration(
            fixed=self.template,
            moving=image,
            initial_transform=rigid_transform,
            type_of_transform=self.type_of_transform,
        )["fwdtransforms"]
        registered = ants.apply_transforms(
            self.template, image, transform, interpolator=self.interpolator,
        )
        return registered.to_nibabel()

    def register_images(self, images: List[NiftiImage]) -> List[NiftiImage]:
        return [self(image) for image in images]

    def register_images_to_templates(
        self, images: List[NiftiImage], templates: List[NiftiImage],
    ) -> List[NiftiImage]:
        registered = []
        original_template = self.template
        for image, template in zip(images, templates):
            self.template = template
            registered.append(self(image))
        self.template = original_template
        return registered
