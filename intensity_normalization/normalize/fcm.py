# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.fcm

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "FCMNormalize",
]

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Optional, Type, TypeVar

import numpy as np

from intensity_normalization import VALID_MODALITIES
from intensity_normalization.type import (
    Array,
    dir_path,
    file_path,
    positive_float,
    save_nifti_path,
)
from intensity_normalization.normalize.base import NormalizeBase, NormalizeDirectoryBase
from intensity_normalization.util.tissue_membership import find_tissue_memberships

FCM = TypeVar("FCM", bound="FCMNormalize")


class FCMNormalize(NormalizeBase):
    """
    use fuzzy c-means-generated tissue membership (found on a T1-w
    image) to normalize the tissue to norm_value (default = 1.)
    """

    tissue_to_int = {"csf": 0, "gm": 1, "wm": 2}

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
            tissue_memberships = find_tissue_memberships(data, mask)
            self.tissue_membership = tissue_memberships[
                ..., self.tissue_to_int[self.tissue_type]
            ]
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
        options = parser.add_argument_group("Options")
        options.add_argument(
            "-p",
            "--plot-histogram",
            action="store_true",
            help="Plot the histogram of the normalized image.",
        )
        options.add_argument(
            "-v",
            "--verbosity",
            action="count",
            default=0,
            help="Increase output verbosity (e.g., -vv is more than -v).",
        )
        return parser

    @staticmethod
    def add_method_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Method")
        parser.add_argument(
            "--tissue-type",
            default="wm",
            type=str,
            choices=("wm", "gm", "csf"),
            help="Reference tissue to use for normalization.",
        )
        group = parent_parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-m",
            "--mask",
            type=file_path(),
            help="Path to a foreground mask for the image. "
            "Provide this if not providing a tissue mask.",
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
        if hasattr(args, "mask"):
            mask = args.mask
            if args.modality is not None:
                if args.modality.lower() != "t1":
                    msg = (
                        "If a brain mask is provided, modality must be `t1`. "
                        f"Got {args.modality}."
                    )
                    raise ValueError(msg)
        else:
            mask = args.tissue_mask
        self.normalize_from_filenames(
            args.image,
            mask,
            args.output,
            args.modality,
        )
