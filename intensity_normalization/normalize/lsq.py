# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.lsq

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "LeastSquaresNormalize",
]

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Type, TypeVar

import nibabel as nib
import numpy as np

from intensity_normalization import VALID_MODALITIES
from intensity_normalization.normalize.base import NormalizeFitBase
from intensity_normalization.type import Array, Vector, dir_path, positive_float
from intensity_normalization.util.io import split_filename
from intensity_normalization.util.tissue_membership import find_tissue_memberships

LSQN = TypeVar("LSQN", bound="LeastSquaresNormalize")

logger = logging.getLogger(__name__)


class LeastSquaresNormalize(NormalizeFitBase):
    def __init__(self, norm_value: float = 1.0):
        super().__init__(norm_value)
        self.tissue_memberships: List[Array] = []

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
        if modality is None:
            modality = "t1"
        if modality == "t1":
            tissue_membership = find_tissue_memberships(data, mask)
            self.tissue_memberships.append(tissue_membership)
        else:
            tissue_membership = mask
        tissue_means = self.tissue_means(data, tissue_membership)
        sf = self.scaling_factor(tissue_means)
        return sf

    def _fit(  # type: ignore[no-untyped-def]
        self,
        images: List[Array],
        masks: Optional[List[Array]] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> None:
        image = images[0]  # only need one image to fit this method
        mask = masks and masks[0]
        assert isinstance(mask, Array) or mask is None
        if modality is None:
            modality = "t1"
        if modality.lower() == "t1":
            tissue_membership = find_tissue_memberships(image, mask)
        else:
            logger.debug("Assuming --mask-dir contains tissue memberships.")
            tissue_membership = mask
        csf_mean = np.average(image, weights=tissue_membership[..., 0])
        norm_image = (image / csf_mean) * self.norm_value
        self.standard_tissue_means = self.tissue_means(
            norm_image,
            tissue_membership,
        )

    @staticmethod
    def tissue_means(image: Array, tissue_membership: Array) -> Vector:
        n_tissues = tissue_membership.shape[-1]
        weighted_avgs = [
            np.average(image, weights=tissue_membership[..., i])
            for i in range(n_tissues)
        ]
        return np.asarray([weighted_avgs]).T

    def scaling_factor(self, tissue_means: Vector) -> float:
        numerator = tissue_means.T @ tissue_means
        denominator = tissue_means.T @ self.standard_tissue_means
        sf: float = (numerator / denominator).item()
        return sf

    @staticmethod
    def name() -> str:
        return "lsq"

    @staticmethod
    def fullname() -> str:
        return "Least Squares"

    @staticmethod
    def description() -> str:
        return (
            "Minimize distance between tissue means (CSF/GM/WM) in a "
            "least squares-sense within a set of NIfTI MR images."
        )

    def save_additional_info(  # type: ignore[no-untyped-def]
        self,
        args: Namespace,
        **kwargs,
    ) -> None:
        for memberships, fn in zip(self.tissue_memberships, kwargs["image_filenames"]):
            tissue_memberships = nib.Nifti1Image(
                memberships,
                None,
            )
            base, name, ext = split_filename(fn)
            new_name = name + "_tissue_memberships" + ext
            if args.output_dir is None:
                output = base / new_name
            else:
                output = Path(args.output_dir) / new_name
            tissue_memberships.to_filename(output)
        del self.tissue_memberships

    @classmethod
    def from_argparse_args(cls: Type[LSQN], args: Namespace) -> LSQN:
        out: LSQN = cls(args.norm_value)
        return out

    def call_from_argparse_args(self, args: Namespace) -> None:
        if args.mask_dir is not None:
            if args.modality is not None:
                if args.modality.lower() != "t1":
                    msg = (
                        "If brain masks provided, modality must be `t1`. "
                        f"Got {args.modality}."
                    )
                    raise ValueError(msg)
        elif args.tissue_membership_dir is not None:
            args.mask_dir = args.tissue_membership_dir
        super().call_from_argparse_args(args)

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
            "-n",
            "--norm-value",
            type=positive_float(),
            default=1.0,
            help="Reference value for normalization.",
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

    @staticmethod
    def add_method_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        exclusive = parent_parser.add_argument_group(
            "mutually exclusive optional arguments"
        )
        group = exclusive.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "-m",
            "--mask-dir",
            type=dir_path(),
            default=None,
            help="Path to a foreground mask for the image. "
            "Provide this if not providing a tissue mask "
            "(if image is not skull-stripped).",
        )
        group.add_argument(
            "-tm",
            "--tissue-membership-dir",
            type=dir_path(),
            help="Path to a mask of a tissue memberships. "
            "Provide this if not providing the foreground mask.",
        )
        return parent_parser
