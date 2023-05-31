"""RAVEL normalization (WhiteStripe then CSF correction)
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: Jun 02, 2021
"""

from __future__ import annotations

__all__ = ["RavelNormalize"]

import argparse
import collections.abc
import functools
import logging
import operator
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import pymedio.image as mioi
import scipy.sparse
import scipy.sparse.linalg

import intensity_normalization.errors as intnorme
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
        membership_threshold: float = 0.99,
        register: bool = True,
        num_unwanted_factors: int = 1,
        sparse_svd: bool = False,
        whitestripe_kwargs: dict[str, typing.Any] | None = None,
        quantile_to_label_csf: float = 1.0,
        masks_are_csf: bool = False,
    ):
        """Normalize a set of co-registered images with WhiteStripe and a CSF correction

        Args:
            membership_threshold: threshold in FCM for CSF membership
            register: if the images aren't deformably co-registered,
                set this to True to do so before calculating the unwanted factors
                NOTE: The images must be rigidly or affine registered, at a minimum,
                before using this process. Good results, however, are only
                reliable for deformably co-registered images!
            num_unwanted_factors: see 'b' from the original paper
            sparse_svd: if you're hitting out-of-memory errors, try setting this to true
            whitestripe_kwargs: keyword args to pass to WhiteStripe
            quantile_to_label_csf: lower this if you want some wiggle room in the number
                of images in which CSF must be found. Don't change this in general.
            masks_are_csf: flag to signify that the masks are actually CSF boolean masks
                so CSF masks are not created, and the CSF masks are used directly.
        """
        super().__init__()
        self.membership_threshold = membership_threshold
        self.register = register
        self.num_unwanted_factors = num_unwanted_factors
        self.sparse_svd = sparse_svd
        self.whitestripe_kwargs = whitestripe_kwargs or dict()
        self.quantile_to_label_csf = quantile_to_label_csf
        self.masks_are_csf = masks_are_csf
        if register and masks_are_csf:
            msg = "If 'masks_are_csf', then images are assumed to be co-registered."
            raise ValueError(msg)
        self._template: intnormt.ImageLike | None = None
        self._template_mask: intnormt.ImageLike | None = None
        self._normalized: intnormt.ImageLike | None = None
        self._control_masks: list[intnormt.ImageLike] = []

    def normalize_image(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> intnormt.ImageLike:
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
        template: intnormt.ImageLike | ants.ANTsImage,
    ) -> None:
        self._template = to_ants(template)

    def set_template_mask(
        self,
        template_mask: intnormt.ImageLike | ants.ANTsImage | None,
    ) -> None:
        if template_mask is None:
            self._template_mask = None
        else:
            if hasattr(template_mask, "astype"):
                template_mask = template_mask.astype(np.uint32)
            self._template_mask = to_ants(template_mask)

    def use_mni_as_template(self) -> None:
        standard_mni = ants.get_ants_data("mni")
        self.set_template(ants.image_read(standard_mni))
        assert self.template is not None
        self.set_template_mask(self.template > 0.0)

    def _find_csf_mask(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> intnormt.ImageLike:
        if self.masks_are_csf:
            if mask is None:
                raise ValueError("'mask' must be defined if masks are CSF masks.")
            return mask
        elif modality != intnormt.Modality.T1:
            msg = "Non-T1-w RAVEL normalization w/o CSF masks not supported."
            raise NotImplementedError(msg)
        tissue_membership = intnormtm.find_tissue_memberships(image, mask)
        csf_mask: npt.NDArray = tissue_membership[..., 0] > self.membership_threshold
        # convert to integer for intersection
        csf_mask = csf_mask.astype(np.uint32)
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
        logger.debug("Performing RAVEL correction.")
        logger.debug(f"Unwanted factors shape: {unwanted_factors.shape}.")
        logger.debug(f"Control voxels shape: {control_voxels.shape}.")
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
        registered: ants.ANTsImage = register(
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
        images: collections.abc.Sequence[intnormt.ImageLike],
        /,
        masks: collections.abc.Sequence[intnormt.ImageLike] | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """creates a matrix of images; rows correspond to voxels, columns are images

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
        if any([shape != image_shape for shape in image_shapes]):
            msg = "All images must be the same size and have (approximate) voxel-wise "
            msg += "correspondence. At a minimum, rigid/affine co-register the images."
            raise RuntimeError(msg)
        image_matrix = np.zeros((image_size, n_images))
        whitestripe_norm = intnormws.WhiteStripeNormalize(**self.whitestripe_kwargs)
        self._control_masks = []  # reset control masks to prevent run-to-run issues
        registered_images = []

        for i, (image, mask) in enumerate(intnormio.zip_with_nones(images, masks), 1):
            image_ws = whitestripe_norm(image, mask)
            # the line below is why the images have to be the same size
            image_matrix[:, i - 1] = image_ws.flatten()
            logger.info(f"Processing image {i}/{n_images}.")
            if i == 1 and self.template is None:
                logger.debug("Setting template to first image.")
                self.set_template(image)
                self.set_template_mask(mask)
                logger.debug("Finding CSF mask.")
                # CSF found on original b/c assume foreground positive
                csf_mask = self._find_csf_mask(image, mask, modality=modality)
                self._control_masks.append(csf_mask)
                if self.register:
                    registered_images.append(image_ws)
            else:
                if self.register:
                    logger.debug("Deformably co-registering image to template.")
                    image = to_ants(image)
                    image = self._register(image)
                    image_ws = whitestripe_norm(image, mask)
                    registered_images.append(image_ws)
                logger.debug("Finding CSF mask.")
                csf_mask = self._find_csf_mask(image, mask, modality=modality)
                self._control_masks.append(csf_mask)

        control_mask_sum = functools.reduce(operator.add, self._control_masks)
        threshold = np.floor(len(self._control_masks) * self.quantile_to_label_csf)
        intersection: intnormt.ImageLike = control_mask_sum >= threshold
        num_control_voxels = int(intersection.sum())
        if num_control_voxels == 0:
            msg = "No common control voxels found. Lower the membership threshold."
            raise intnorme.NormalizationError(msg)
        if self.register:
            assert n_images == len(registered_images)
            control_voxels = np.zeros((num_control_voxels, n_images))
            for i, registered in enumerate(registered_images):
                ctrl_vox = registered[intersection]
                control_voxels[:, i] = ctrl_vox
                logger.debug(
                    f"Image {i+1} control voxels - "
                    f"mean: {ctrl_vox.mean():.3f}; "
                    f"std: {ctrl_vox.std():.3f}"
                )
        else:
            control_voxels = image_matrix[intersection.flatten(), :]

        return image_matrix, control_voxels

    def estimate_unwanted_factors(self, control_voxels: npt.NDArray) -> npt.NDArray:
        logger.debug("Estimating unwanted factors.")
        _, _, all_unwanted_factors = (
            np.linalg.svd(control_voxels, full_matrices=False)
            if not self.sparse_svd
            else scipy.sparse.linalg.svds(
                scipy.sparse.bsr_matrix(control_voxels),
                k=self.num_unwanted_factors,
                return_singular_vectors="vh",
            )
        )
        unwanted_factors: npt.NDArray
        unwanted_factors = all_unwanted_factors.T[:, 0 : self.num_unwanted_factors]
        return unwanted_factors

    def _fit(
        self,
        images: collections.abc.Sequence[intnormt.ImageLike],
        /,
        masks: collections.abc.Sequence[intnormt.ImageLike] | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
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
        modality: intnormt.Modality = intnormt.Modality.T1,
        ext: str = "nii*",
        return_normalized_and_masks: bool = False,
        **kwargs: typing.Any,
    ) -> tuple[list[mioi.Image], list[mioi.Image] | None] | None:
        logger.debug("Gathering images.")
        images, masks = intnormio.gather_images_and_masks(image_dir, mask_dir, ext=ext)
        self.fit(images, masks, modality=modality, **kwargs)
        assert self._normalized is not None
        if return_normalized_and_masks:
            norm_lst: list[mioi.Image] = []
            for normed, image in zip(self._normalized, images):
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
        desc += "variation with RAVEL on a set of MR images."
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
            help="Number of unwanted factors to eliminate (see 'b' in RAVEL paper).",
        )
        parser.add_argument(
            "-mt",
            "--membership-threshold",
            type=float,
            default=0.99,
            help="Threshold for the membership of the control (CSF) voxels.",
        )
        parser.add_argument(
            "--no-registration",
            action="store_false",
            dest="register",
            default=True,
            help="Do not do deformable registration to find control mask. "
            "(Assumes images are deformably co-registered).",
        )
        parser.add_argument(
            "--sparse-svd",
            action="store_true",
            default=False,
            help="Use a sparse version of the SVD (lower memory requirements).",
        )
        parser.add_argument(
            "--masks-are-csf",
            action="store_true",
            default=False,
            help="Use this flag if mask directory corresponds to CSF masks "
            "instead of brain masks. (Assumes images are deformably co-registered).",
        )
        parser.add_argument(
            "--quantile-to-label-csf",
            default=1.0,
            help="Control how intersection calculated "
            "(1.0 means normal intersection, 0.5 means only "
            "half of the images need the voxel labeled as CSF).",
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

    def save_additional_info(
        self,
        args: argparse.Namespace,
        **kwargs: typing.Any,
    ) -> None:
        normed = kwargs["normalized"]
        image_fns = kwargs["image_filenames"]
        if len(self._control_masks) != len(image_fns):
            msg = f"'control_masks' ({len(self._control_masks)}) "
            msg += f"and 'image_filenames' ({len(image_fns)}) "
            msg += "must be in correspondence."
            raise RuntimeError(msg)
        if len(self._control_masks) != len(normed):
            msg = f"'control_masks' ({len(self._control_masks)}) "
            msg += f"and 'normalized' ({len(normed)}) "
            msg += "must be in correspondence."
            raise RuntimeError(msg)
        for _csf_mask, norm, fn in zip(self._control_masks, normed, image_fns):
            csf_mask: mioi.Image
            if hasattr(norm, "affine"):
                csf_mask = mioi.Image(_csf_mask, norm.affine)
            elif hasattr(_csf_mask, "affine"):
                csf_mask = mioi.Image(_csf_mask, _csf_mask.affine)
            else:
                csf_mask = mioi.Image(_csf_mask, None)
            base, name, ext = intnormio.split_filename(fn)
            new_name = name + "_csf_mask" + ext
            if args.output_dir is None:
                output = base / new_name
            else:
                output = pathlib.Path(args.output_dir) / new_name
            csf_mask.to_filename(output)
        del self._control_masks
        self._control_masks = []
