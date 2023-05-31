"""Fuzzy C-Means-based tissue mean normalization
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["FCMNormalize"]

import argparse
import logging
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import pymedio.image as mioi

import intensity_normalization as intnorm
import intensity_normalization.normalize.base as intnormb
import intensity_normalization.typing as intnormt
import intensity_normalization.util.io as intnormio
import intensity_normalization.util.tissue_membership as intnormtm

logger = logging.getLogger(__name__)


class FCMNormalize(intnormb.LocationScaleCLIMixin, intnormb.SingleImageNormalizeCLI):
    def __init__(
        self,
        *,
        norm_value: float = 1.0,
        tissue_type: intnormt.TissueType = intnormt.TissueType.WM,
        **kwargs: typing.Any,
    ):
        """
        Use fuzzy c-means-generated tissue membership (found on a T1-w image) to
        normalize the specified tissue type's mean to norm_value (default = 1.)
        """
        super().__init__(norm_value=norm_value, **kwargs)
        self.tissue_membership: npt.NDArray | None = None
        self.tissue_type = tissue_type

    def calculate_location(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> float:
        return 0.0

    def calculate_scale(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> float:
        tissue_mean: float
        if modality == intnormt.Modality.T1:
            mask = self._get_mask(image, mask, modality=modality)
            tissue_name = self.tissue_type.to_fullname()
            logger.debug(f"Finding {tissue_name} membership.")
            tissue_memberships = intnormtm.find_tissue_memberships(image, mask)
            self.tissue_membership = tissue_memberships[..., self.tissue_type.to_int()]
            logger.debug(f"Calculated {tissue_name} membership.")
            tissue_mean = float(np.average(image, weights=self.tissue_membership))
        elif modality != intnormt.Modality.T1 and mask is None and self.is_fit:
            tissue_mean = float(np.average(image, weights=self.tissue_membership))
        elif modality != intnormt.Modality.T1 and mask is not None:
            tissue_mean = float(np.average(image, weights=mask))
        else:
            msg = "Either a T1-w image must be passed to initialize a tissue "
            msg += "membership mask or the tissue memberships must be provided."
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
        desc = "Use fuzzy c-means to find memberships of CSF/GM/WM in the brain. "
        desc += "Use the specified tissue's mean to normalize a MRI."
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
            "-o",
            "--output",
            type=intnormt.save_file_path(),
            default=None,
            help="Path to save normalized image.",
        )
        parser.add_argument(
            "-mo",
            "--modality",
            type=str,
            default="t1",
            choices=valid_modalities,
            help="Modality of the image.",
        )
        parser.add_argument(
            "-n",
            "--norm-value",
            type=intnormt.positive_float(),
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
        parser.add_argument(
            "--version",
            action="store_true",
            help="Print the version of intensity-normalization.",
        )
        return parser

    @staticmethod
    def add_method_specific_arguments(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
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
            type=intnormt.file_path(),
            help="Path to a foreground mask for the image. "
            "Provide this if not providing a tissue mask. "
            "(If image is not skull-stripped, this is required.)",
        )
        group.add_argument(
            "-tm",
            "--tissue-mask",
            type=intnormt.file_path(),
            help="Path to a mask of a target tissue (usually found through FCM). "
            "Provide this if not providing the foreground mask.",
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace, /) -> FCMNormalize:
        tt = intnormt.TissueType.from_string(args.tissue_type)
        return cls(norm_value=args.norm_value, tissue_type=tt)

    def call_from_argparse_args(
        self, args: argparse.Namespace, /, **kwargs: typing.Any
    ) -> None:
        if args.mask is not None:
            if args.modality is not None:
                if args.modality.lower() != "t1":
                    msg = "If a brain mask is provided, 'modality' must be 't1'. "
                    msg += f"Got '{args.modality}'."
                    raise ValueError(msg)
        elif args.tissue_mask is not None:
            args.mask = args.tissue_mask
        super().call_from_argparse_args(args)

    def save_additional_info(
        self,
        args: argparse.Namespace,
        **kwargs: typing.Any,
    ) -> None:
        if self.is_fit and args.tissue_mask is None:
            assert self.tissue_membership is not None
            tissue_membership: mioi.Image = mioi.Image(
                self.tissue_membership,
                kwargs["normalized"].affine,
            )
            base, name, ext = intnormio.split_filename(args.image)
            new_name = name + f"_{self.tissue_type.value}_membership" + ext
            if args.output is None:
                output = base / new_name
            else:
                output = pathlib.Path(args.output).parent / new_name
            logger.info(f"Saving tissue membership: {output}")
            tissue_membership.to_filename(output)
