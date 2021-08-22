# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.ravel

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 02, 2021
"""

__all__ = [
    "RavelNormalize",
]

import logging
from argparse import ArgumentParser, Namespace
from functools import reduce
from operator import add
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import nibabel
import numpy as np
from numpy.linalg import solve
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import svds

from intensity_normalization.normalize.base import NormalizeFitBase
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.type import Array, ArrayOrNifti, NiftiImage, PathLike
from intensity_normalization.util.coregister import register, to_ants
from intensity_normalization.util.io import gather_images_and_masks
from intensity_normalization.util.tissue_membership import find_tissue_memberships

logger = logging.getLogger(__name__)

try:
    import ants
except (ModuleNotFoundError, ImportError):
    logger.error("ANTsPy not installed. Install antspyx to use RAVEL.")
    raise

RN = TypeVar("RN", bound="RavelNormalize")


class RavelNormalize(NormalizeFitBase):
    def __init__(
        self,
        membership_threshold: float = 0.99,
        register: bool = True,
        num_unwanted_factors: int = 1,
        sparse_svd: bool = False,
        whitestripe_kwargs: Optional[Dict[str, Any]] = None,
        proportion_of_intersection_to_label_csf: float = 1.0,
        masks_are_csf: bool = False,
    ):
        super().__init__()
        self.membership_threshold = membership_threshold
        self.register = register
        self.num_unwanted_factors = num_unwanted_factors
        self.sparse_svd = sparse_svd
        self.whitestripe_kwargs = whitestripe_kwargs or dict()
        self.proportion_of_intersection_to_label_csf = (
            proportion_of_intersection_to_label_csf
        )
        self.masks_are_csf = masks_are_csf
        if register and masks_are_csf:
            raise ValueError(
                "If masks_are_csf, then images are assumed to be co-registered."
            )
        self._template = None
        self._template_mask = None

    def calculate_location(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> float:
        return 0.0

    def calculate_scale(
        self,
        data: Array,
        mask: Optional[Array] = None,
        modality: Optional[str] = None,
    ) -> float:
        return 1.0

    @property
    def template(self) -> Optional[ants.ANTsImage]:
        return self._template

    @property
    def template_mask(self) -> Optional[ants.ANTsImage]:
        return self._template_mask

    def set_template(
        self,
        template: Union[ArrayOrNifti, ants.ANTsImage],
    ) -> None:
        self._template = to_ants(template)

    def set_template_mask(
        self,
        template_mask: Optional[Union[ArrayOrNifti, ants.ANTsImage]],
    ) -> None:
        if template_mask is None:
            self._template_mask = None
        else:
            self._template_mask = to_ants(template_mask)

    def use_mni_as_template(self) -> None:
        standard_mni = ants.get_ants_data("mni")
        self.set_template(ants.image_read(standard_mni))
        assert self.template is not None
        self.set_template_mask(self.template > 0.0)

    def _find_csf_mask(
        self,
        image: Array,
        mask: Optional[Array],
        modality: Optional[str] = None,
    ) -> Array:
        if self.masks_are_csf:
            assert mask is not None
            return mask
        elif self._get_modality(modality) != "t1":
            raise NotImplementedError(
                "Non-T1-w RAVEL normalization w/o CSF masks not supported."
            )
        tissue_membership = find_tissue_memberships(image, mask)
        csf_mask: Array = tissue_membership[..., 0] > self.membership_threshold
        csf_mask = csf_mask.astype(np.uint8)  # convert to integer for intersection
        return csf_mask

    @staticmethod
    def _ravel_correction(control_voxels: Array, unwanted_factors: Array) -> Array:
        """Correct control voxels by removing trend from unwanted factors

        Args:
            control_voxels: rows are voxels, columns are images
                (see V matrix in the paper)
            unwanted_factors: unwanted factors
                (see Z matrix in the paper)

        Returns:
            normalized: normalized images
        """
        logger.debug("Performing RAVEL correction")
        logger.debug(f"Unwanted factors shape: {unwanted_factors.shape}")
        logger.debug(f"Control voxels shape: {control_voxels.shape}")
        beta = solve(
            unwanted_factors.T @ unwanted_factors,
            unwanted_factors.T @ control_voxels.T,
        )
        fitted = (unwanted_factors @ beta).T
        residuals = control_voxels - fitted
        voxel_means = np.mean(control_voxels, axis=1, keepdims=True)
        normalized: Array = residuals + voxel_means
        return normalized

    def _register(self, image: ants.ANTsImage) -> Array:
        registered = register(
            image=image,
            template=self.template,
            type_of_transform="SyN",
            interpolator="linear",
            template_mask=self.template_mask,
        )
        out: Array = registered.numpy()
        return out

    def create_image_matrix_and_control_voxels(
        self,
        images: List[Array],
        masks: Optional[List[Optional[Array]]] = None,
        modality: Optional[str] = None,
    ) -> Tuple[Array, Array]:
        """creates an matrix of images; rows correspond to voxels, columns are images

        Args:
            images: list of MR images of interest
            masks: list of corresponding brain masks
            modality: modality of the set of images (e.g., t1)

        Returns:
            image_matrix: rows are voxels, columns are images
            control_voxels: rows are csf intersection voxels, columns are images
        """
        n_images = len(images)
        image_shapes = [image.shape for image in images]
        image_shape = image_shapes[0]
        image_size = int(np.prod(image_shape))
        assert all([shape == image_shape for shape in image_shapes])
        image_matrix = np.zeros((image_size, n_images))
        whitestripe_norm = WhiteStripeNormalize(**self.whitestripe_kwargs)
        control_masks = []
        registered_images = []
        masks = ([None] * n_images) if masks is None else masks
        assert n_images == len(masks)

        for i, (image, mask) in enumerate(zip(images, masks), 1):
            image_ws = whitestripe_norm(image)
            image_matrix[:, i - 1] = image_ws.flatten()
            logger.info(f"Processing image {i}/{n_images}")
            if i == 1 and self.template is None:
                logger.debug("Setting template to first image")
                self.set_template(image)
                self.set_template_mask(mask)
                logger.debug("Finding CSF mask")
                # csf found on original b/c assume foreground positive
                csf_mask = self._find_csf_mask(image, mask, modality)
                control_masks.append(csf_mask)
                if self.register:
                    registered_images.append(image_ws)
            else:
                if self.register:
                    logger.debug("Deformably co-registering image to template")
                    image = to_ants(image)
                    image = self._register(image)
                    image_ws = whitestripe_norm(image)
                    registered_images.append(image_ws)
                logger.debug("Finding CSF mask")
                csf_mask = self._find_csf_mask(image, mask, modality)
                control_masks.append(csf_mask)

        control_mask_sum = reduce(add, control_masks)
        threshold = np.floor(
            len(control_masks) * self.proportion_of_intersection_to_label_csf
        )
        intersection = control_mask_sum >= threshold
        num_control_voxels = int(intersection.sum())
        if num_control_voxels == 0:
            raise RuntimeError(
                "No common control voxels were found. "
                "Lower the membership threshold."
            )
        if self.register:
            assert n_images == len(registered_images)
            control_voxels = np.zeros((num_control_voxels, n_images))
            for i, registered in enumerate(registered_images):
                ctrl_vox = registered[intersection]
                control_voxels[:, i] = ctrl_vox
                logger.info(
                    f"Image {i+1} control voxels - "
                    f"mean: {ctrl_vox.mean():.3f}; "
                    f"std: {ctrl_vox.std():.3f}"
                )
        else:
            control_voxels = image_matrix[intersection.flatten(), :]

        return image_matrix, control_voxels

    def estimate_unwanted_factors(self, control_voxels: Array) -> Array:
        logger.debug("Estimating unwanted factors")
        _, _, all_unwanted_factors = (
            np.linalg.svd(control_voxels, full_matrices=False)
            if not self.sparse_svd
            else svds(
                bsr_matrix(control_voxels),
                k=self.num_unwanted_factors,
                return_singular_vectors="vh",
            )
        )
        unwanted_factors: Array = all_unwanted_factors.T[
            :, 0 : self.num_unwanted_factors
        ]
        return unwanted_factors

    def fit(  # type: ignore[no-untyped-def,override]
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> Array:
        images, masks = self.before_fit(images, masks, modality, **kwargs)
        logger.info("Fitting")
        normalized = self._fit(images, masks, modality, **kwargs)
        logger.debug("Done fitting")
        return normalized

    def _fit(  # type: ignore[no-untyped-def,override]
        self,
        images: List[ArrayOrNifti],
        masks: Optional[List[ArrayOrNifti]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> Array:
        image_matrix, control_voxels = self.create_image_matrix_and_control_voxels(
            images,
            masks,
            modality,
        )
        unwanted_factors = self.estimate_unwanted_factors(control_voxels)
        normalized = self._ravel_correction(image_matrix, unwanted_factors)
        return normalized.T  # transpose so images on 0th axis

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
        normalized = self.fit(images, masks, modality, **kwargs)
        if return_normalized_and_masks:
            normalized_list: List[NiftiImage] = []
            for normed, image in zip(normalized, images):
                shape = image.get_fdata().shape
                normalized_list.append(
                    nibabel.Nifti1Image(
                        normed.reshape(shape),
                        image.affine,
                        image.header,
                    )
                )
            return normalized_list, masks
        return None

    @staticmethod
    def name() -> str:
        return "ravel"

    @staticmethod
    def fullname() -> str:
        return "RAVEL"

    @staticmethod
    def description() -> str:
        return (
            "Perform WhiteStripe and then correct for technical "
            "variation with RAVEL on a set of NIfTI MR images."
        )

    @staticmethod
    def add_method_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("method-specific arguments")
        parser.add_argument(
            "-b",
            "--num-unwanted-factors",
            type=int,
            default=1,
            help="number of unwanted factors to eliminate (see b in RAVEL paper)",
        )
        parser.add_argument(
            "-mt",
            "--membership-threshold",
            type=float,
            default=0.99,
            help="threshold for the membership of the control (CSF) voxels",
        )
        parser.add_argument(
            "--no-registration",
            action="store_false",
            dest="register",
            default=True,
            help="do not do deformable registration to find control mask",
        )
        parser.add_argument(
            "--sparse-svd",
            action="store_true",
            default=False,
            help="use a sparse version of the svd (lower memory requirements)",
        )
        parser.add_argument(
            "--masks-are-csf",
            action="store_true",
            default=False,
            help="mask directory corresponds to csf masks instead of brain masks, "
            "assumes images are deformably co-registered",
        )
        parser.add_argument(
            "--prop-intersection-csf",
            default=1.0,
            help="control how intersection calculated "
            "(1.0 means normal intersection, 0.5 means only "
            "half of the images need the voxel labeled as csf)",
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls: Type[RN], args: Namespace) -> RN:
        return cls(
            membership_threshold=args.membership_threshold,
            register=args.register,
            num_unwanted_factors=args.num_unwanted_factors,
            sparse_svd=args.sparse_svd,
            proportion_of_intersection_to_label_csf=args.prop_intersection_csf,
            masks_are_csf=args.masks_are_csf,
        )
