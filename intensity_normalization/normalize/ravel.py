"""RAVEL normalization (WhiteStripe then CSF correction)
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: Jun 02, 2021
"""

from __future__ import annotations

__all__ = ["RavelNormalize"]

import argparse
import builtins
import functools
import logging
import operator
import typing

import numpy as np
import numpy.typing as npt
import pymedio.image as mioi
import scipy.sparse
import scipy.sparse.linalg

import intensity_normalization.normalize.base as intnormb
import intensity_normalization.normalize.whitestripe as intnormws
import intensity_normalization.typing as intnormt
import intensity_normalization.util.io as intnormio
import intensity_normalization.util.tissue_membership as intnormtm

logger = logging.getLogger(__name__)

try:
    import ants
except ImportError as ants_imp_exn:
    msg = "ANTsPy not installed. Install antspyx to use RAVEL."
    raise RuntimeError(msg) from ants_imp_exn
else:
    from intensity_normalization.util.coregister import register, to_ants


class RavelNormalize(intnormb.DirectoryNormalizeCLI):
    def __init__(
        self,
        *,
        membership_threshold: builtins.float = 0.99,
        register: builtins.bool = True,
        num_unwanted_factors: builtins.int = 1,
        sparse_svd: builtins.bool = False,
        whitestripe_kwargs: typing.Dict[builtins.str, typing.Any] | None = None,
        quantile_to_label_csf: builtins.float = 1.0,
        masks_are_csf: builtins.bool = False,
    ):
        super().__init__()
        self.membership_threshold = membership_threshold
        self.register = register
        self.num_unwanted_factors = num_unwanted_factors
        self.sparse_svd = sparse_svd
        self.whitestripe_kwargs = whitestripe_kwargs or dict()
        self.quantile_to_label_csf = quantile_to_label_csf
        self.masks_are_csf = masks_are_csf
        if register and masks_are_csf:
            msg = "If masks_are_csf, then images are assumed to be co-registered."
            raise ValueError(msg)
        self._template: intnormt.Image | None = None
        self._template_mask: intnormt.Image | None = None
        self._normalized: intnormt.Image | None = None

    def normalize_image(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> intnormt.Image:
        return NotImplemented

    def teardown(self) -> None:
        del self._normalized
        self._normalized = None

    @property
    def template(self) -> ants.ANTsImage | None:
        return self._template

    @property
    def template_mask(self) -> ants.ANTsImage | None:
        return self._template_mask

    def set_template(
        self,
        template: intnormt.Image | ants.ANTsImage,
    ) -> None:
        self._template = to_ants(template)

    def set_template_mask(
        self,
        template_mask: intnormt.Image | ants.ANTsImage | None,
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
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> intnormt.Image:
        if self.masks_are_csf:
            if mask is None:
                raise ValueError("mask must be defined if masks are CSF masks.")
            return mask
        elif modality != intnormt.Modalities.T1:
            msg = "Non-T1-w RAVEL normalization w/o CSF masks not supported."
            raise NotImplementedError(msg)
        tissue_membership = intnormtm.find_tissue_memberships(image, mask)
        csf_mask: npt.NDArray = tissue_membership[..., 0] > self.membership_threshold
        csf_mask = csf_mask.astype(np.uint8)  # convert to integer for intersection
        return csf_mask

    @staticmethod
    def _ravel_correction(
        control_voxels: npt.NDArray, unwanted_factors: npt.NDArray
    ) -> npt.NDArray:
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
        beta = np.linalg.solve(
            unwanted_factors.T @ unwanted_factors,
            unwanted_factors.T @ control_voxels.T,
        )
        fitted = (unwanted_factors @ beta).T
        residuals: np.ndarray = control_voxels - fitted
        voxel_means = np.mean(control_voxels, axis=1, keepdims=True)
        normalized: npt.NDArray = residuals + voxel_means
        return normalized

    def _register(self, image: ants.ANTsImage) -> npt.NDArray:
        registered = register(
            image,
            template=self.template,
            type_of_transform="SyN",
            interpolator="linear",
            template_mask=self.template_mask,
        )
        out: npt.NDArray = registered.numpy()
        return out

    def create_image_matrix_and_control_voxels(
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
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
        whitestripe_norm = intnormws.WhiteStripeNormalize(**self.whitestripe_kwargs)
        control_masks = []
        registered_images = []

        for i, (image, mask) in enumerate(intnormio.zip_with_nones(images, masks), 1):
            image_ws = whitestripe_norm(image, mask)
            image_matrix[:, i - 1] = image_ws.flatten()
            logger.info(f"Processing image {i}/{n_images}")
            if i == 1 and self.template is None:
                logger.debug("Setting template to first image")
                self.set_template(image)
                self.set_template_mask(mask)
                logger.debug("Finding CSF mask")
                # csf found on original b/c assume foreground positive
                csf_mask = self._find_csf_mask(image, mask, modality=modality)
                control_masks.append(csf_mask)
                if self.register:
                    registered_images.append(image_ws)
            else:
                if self.register:
                    logger.debug("Deformably co-registering image to template")
                    image = to_ants(image)
                    image = self._register(image)
                    image_ws = whitestripe_norm(image, mask)
                    registered_images.append(image_ws)
                logger.debug("Finding CSF mask")
                csf_mask = self._find_csf_mask(image, mask, modality=modality)
                control_masks.append(csf_mask)

        control_mask_sum = functools.reduce(operator.add, control_masks)
        threshold = np.floor(len(control_masks) * self.quantile_to_label_csf)
        intersection: intnormt.Image = control_mask_sum >= threshold
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

    def estimate_unwanted_factors(self, control_voxels: npt.NDArray) -> npt.NDArray:
        logger.debug("Estimating unwanted factors")
        _, _, all_unwanted_factors = (
            np.linalg.svd(control_voxels, full_matrices=False)
            if not self.sparse_svd
            else scipy.sparse.linalg.svds(
                scipy.sparse.bsr_matrix(control_voxels),
                k=self.num_unwanted_factors,
                return_singular_vectors="vh",
            )
        )
        unwanted_factors: npt.NDArray = all_unwanted_factors.T[
            :, 0 : self.num_unwanted_factors
        ]
        return unwanted_factors

    def _fit(
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
        **kwargs: typing.Any,
    ) -> None:
        image_matrix, control_voxels = self.create_image_matrix_and_control_voxels(
            images,
            masks,
            modality=modality,
        )
        unwanted_factors = self.estimate_unwanted_factors(control_voxels)
        normalized = self._ravel_correction(image_matrix, unwanted_factors)
        self._normalized = normalized.T  # transpose so images on 0th axis

    def process_directories(
        self,
        image_dir: intnormt.PathLike,
        /,
        mask_dir: intnormt.PathLike | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
        ext: builtins.str = "nii*",
        return_normalized_and_masks: builtins.bool = False,
        **kwargs: typing.Any,
    ) -> typing.Tuple[typing.List[mioi.Image], typing.List[mioi.Image] | None] | None:
        logger.debug("Grabbing images")
        images, masks = intnormio.gather_images_and_masks(image_dir, mask_dir, ext=ext)
        self.fit(images, masks, modality=modality, **kwargs)
        assert self._normalized is not None
        if return_normalized_and_masks:
            norm_lst: typing.List[mioi.Image] = []
            for normed, image in zip(self._normalized, images):  # type: ignore[call-overload] # noqa: E501
                norm_lst.append(mioi.Image(normed.reshape(image.shape), image.affine))
            return norm_lst, masks
        return None

    @staticmethod
    def name() -> str:
        return "ravel"

    @staticmethod
    def fullname() -> str:
        return "RAVEL"

    @staticmethod
    def description() -> str:
        desc = "Perform WhiteStripe and then correct for technical "
        desc += "variation with RAVEL on a set of NIfTI MR images."
        return desc

    @staticmethod
    def add_method_specific_arguments(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
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
            "--quantile-to-label-csf",
            default=1.0,
            help="control how intersection calculated "
            "(1.0 means normal intersection, 0.5 means only "
            "half of the images need the voxel labeled as csf)",
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace, /) -> RavelNormalize:
        return cls(
            membership_threshold=args.membership_threshold,
            register=args.register,
            num_unwanted_factors=args.num_unwanted_factors,
            sparse_svd=args.sparse_svd,
            quantile_to_label_csf=args.quantile_to_label_csf,
            masks_are_csf=args.masks_are_csf,
        )
