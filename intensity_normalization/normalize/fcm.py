# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.fcm

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "FCMNormalize",
]

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Type, TypeVar

import nibabel as nib
import numpy as np

from intensity_normalization import VALID_MODALITIES
from intensity_normalization.normalize.base import NormalizeBase
from intensity_normalization.type import (
    Array,
    file_path,
    positive_float,
    save_nifti_path,
)
from intensity_normalization.util.io import split_filename
from intensity_normalization.util.tissue_membership import find_tissue_memberships

FCM = TypeVar("FCM", bound="FCMNormalize")

logger = logging.getLogger(__name__)


class FCMNormalize(NormalizeBase):
    """
    use fuzzy c-means-generated tissue membership (found on a T1-w
    image) to normalize the tissue to norm_value (default = 1.)
    """

    tissue_to_int = {"csf": 0, "gm": 1, "wm": 2}
    tissue_to_fullname = {"csf": "CSF", "gm": "grey matter", "wm": "white matter"}

    def __init__(self, norm_value: float = 1.0, tissue_type: str = "wm"):
        super().__init__(norm_value)
        self.tissue_membership = None
        self.tissue_type = tissue_type

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
        modality = self._get_modality(modality)
        tissue_mean: float
        if modality == "t1":
            mask = self._get_mask(data, mask, modality)
            tissue_name = self.tissue_to_fullname[self.tissue_type]
            logger.debug(f"Finding {tissue_name} membership")
            tissue_memberships = find_tissue_memberships(data, mask)
            self.tissue_membership = tissue_memberships[
                ..., self.tissue_to_int[self.tissue_type]
            ]
            logger.debug(f"Calculated {tissue_name} membership")
            tissue_mean = np.average(data, weights=self.tissue_membership)
        elif modality != "t1" and mask is None and self.is_fit:
            tissue_mean = np.average(data, weights=self.tissue_membership)
        elif modality != "t1" and mask is not None:
            tissue_mean = np.average(data, weights=mask)
        else:
            msg = (
                "Either a T1-w image must be passed to initialize a tissue "
                "membership mask or the tissue memberships must be provided."
            )
            raise ValueError(msg)
        return tissue_mean

    @property
    def is_fit(self) -> bool:
        return self.tissue_membership is not None

    @staticmethod
    def name() -> str:
        return "fcm"

    @staticmethod
    def fullname() -> str:
        return "Fuzzy C-Means"

    @staticmethod
    def description() -> str:
        return (
            "Use fuzzy c-means to find memberships of CSF/GM/WM in the brain. "
            "Use the found and specified tissue mean to normalize a NIfTI MRI."
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
            "-o",
            "--output",
            type=save_nifti_path(),
            default=None,
            help="Path to save normalized image.",
        )
        parser.add_argument(
            "-mo",
            "--modality",
            type=str,
            default=None,
            choices=VALID_MODALITIES,
            help="Modality of the image.",
        )
        parser.add_argument(
            "-n",
            "--norm-value",
            type=positive_float(),
            default=1.0,
            help="Reference value for normalization.",
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
        parser = parent_parser.add_argument_group("method-specific arguments")
        parser.add_argument(
            "-tt",
            "--tissue-type",
            default="wm",
            type=str,
            choices=("wm", "gm", "csf"),
            help="Reference tissue to use for normalization.",
        )
        exclusive = parent_parser.add_argument_group(
            "mutually exclusive optional arguments"
        )
        group = exclusive.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "-m",
            "--mask",
            type=file_path(),
            help="Path to a foreground mask for the image. "
            "Provide this if not providing a tissue mask "
            "(if image is not skull-stripped).",
        )
        group.add_argument(
            "-tm",
            "--tissue-mask",
            type=file_path(),
            help="Path to a mask of a target tissue (usually found through FCM). "
            "Provide this if not providing the foreground mask.",
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls: Type[FCM], args: Namespace) -> FCM:
        return cls(args.norm_value, args.tissue_type)

    def call_from_argparse_args(self, args: Namespace) -> None:
        if args.mask is not None:
            if args.modality is not None:
                if args.modality.lower() != "t1":
                    msg = (
                        "If a brain mask is provided, modality must be `t1`. "
                        f"Got {args.modality}."
                    )
                    raise ValueError(msg)
        elif args.tissue_mask is not None:
            args.mask = args.tissue_mask
        super().call_from_argparse_args(args)

    def save_additional_info(  # type: ignore[no-untyped-def]
        self,
        args: Namespace,
        **kwargs,
    ) -> None:
        if self.is_fit and args.tissue_mask is None:
            tissue_membership = nib.Nifti1Image(
                self.tissue_membership,
                kwargs["normalized"].affine,
                kwargs["normalized"].header,
            )
            base, name, ext = split_filename(args.image)
            new_name = name + f"_{self.tissue_type}_membership" + ext
            if args.output is None:
                output = base / new_name
            else:
                output = Path(args.output).parent / new_name
            logger.info(f"Saving tissue membership: {output}")
            tissue_membership.to_filename(output)
