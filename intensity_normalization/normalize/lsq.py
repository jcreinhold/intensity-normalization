"""Least-squares fit tissue means of a set of images
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["LeastSquaresNormalize"]

import argparse
import builtins
import collections.abc
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


class LeastSquaresNormalize(
    intnormb.LocationScaleCLIMixin, intnormb.DirectoryNormalizeCLI
):
    def __init__(self, *, norm_value: float = 1.0, **kwargs: typing.Any):
        super().__init__(norm_value=norm_value, **kwargs)
        self.tissue_memberships: builtins.list[mioi.Image] = []
        self.standard_tissue_means: npt.NDArray | None = None

    def calculate_location(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> float:
        return 0.0

    def calculate_scale(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> float:
        tissue_membership: intnormt.ImageLike
        if modality == intnormt.Modalities.T1:
            tissue_membership = intnormtm.find_tissue_memberships(image, mask)
            self.tissue_memberships.append(tissue_membership)
        elif mask is not None:
            tissue_membership = self._fix_tissue_membership(image, mask)
        else:
            msg = "If 'modality' != 't1', you must provide a "
            msg += "tissue membership array in the mask argument."
            raise ValueError(msg)
        tissue_means = self.tissue_means(image, tissue_membership)
        sf = self.scaling_factor(tissue_means)
        return sf

    def _fit(
        self,
        images: collections.abc.Sequence[intnormt.ImageLike],
        /,
        masks: collections.abc.Sequence[intnormt.ImageLike] | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
        **kwargs: typing.Any,
    ) -> None:
        image = images[0]  # only need one image to fit this method
        mask = masks[0] if masks is not None else None
        tissue_membership: intnormt.ImageLike
        if not isinstance(mask, np.ndarray) and mask is not None:
            raise ValueError("Mask must be either none or be like a numpy array.")
        if modality == intnormt.Modalities.T1:
            tissue_membership = intnormtm.find_tissue_memberships(image, mask)
        elif mask is not None:
            logger.debug("Assuming 'masks' contains tissue memberships.")
            tissue_membership = self._fix_tissue_membership(image, mask)
        else:
            msg = "If 'modality' != 't1', you must provide a "
            msg += "tissue membership array in the mask argument."
            raise ValueError(msg)
        csf_mean = np.average(image, weights=tissue_membership[..., 0])
        norm_image: intnormt.ImageLike = (image / csf_mean) * self.norm_value
        self.standard_tissue_means = self.tissue_means(
            norm_image,
            tissue_membership,
        )

    def _fix_tissue_membership(
        self, image: intnormt.ImageLike, tissue_membership: intnormt.ImageLike
    ) -> intnormt.ImageLike:
        if tissue_membership.shape[: int(image.ndim)] != image.shape:
            # try to swap last axes b/c sitk, if still doesn't match then fail
            tissue_membership = typing.cast(
                intnormt.ImageLike, np.swapaxes(tissue_membership, -2, -1)
            )
        if tissue_membership.shape[: int(image.ndim)] != image.shape:
            msg = "If masks provided, need to have same spatial shape as image"
            raise RuntimeError(msg)
        return tissue_membership

    @staticmethod
    def tissue_means(
        image: intnormt.ImageLike, /, tissue_membership: intnormt.ImageLike
    ) -> npt.NDArray:
        n_tissues = tissue_membership.shape[-1]
        weighted_avgs = [
            np.average(image, weights=tissue_membership[..., i])
            for i in range(n_tissues)
        ]
        return np.asarray([weighted_avgs]).T

    def scaling_factor(self, tissue_means: npt.NDArray) -> builtins.float:
        numerator = tissue_means.T @ tissue_means
        denominator = tissue_means.T @ self.standard_tissue_means
        sf: builtins.float = (numerator / denominator).item()
        return sf

    @staticmethod
    def name() -> builtins.str:
        return "lsq"

    @staticmethod
    def fullname() -> builtins.str:
        return "Least Squares"

    @staticmethod
    def description() -> str:
        desc = "Minimize distance between tissue means (CSF/GM/WM) in a "
        desc += "least squares-sense within a set of MR images."
        return desc

    def save_additional_info(
        self,
        args: argparse.Namespace,
        **kwargs: typing.Any,
    ) -> None:
        for memberships, fn in zip(self.tissue_memberships, kwargs["image_filenames"]):
            tissue_memberships = mioi.Image(memberships)
            base, name, ext = intnormio.split_filename(fn)
            new_name = name + "_tissue_memberships" + ext
            if args.output_dir is None:
                output = base / new_name
            else:
                output = pathlib.Path(args.output_dir) / new_name
            tissue_memberships.save(output, squeeze=False)
        del self.tissue_memberships
        if args.save_standard_tissue_means is not None:
            self.save_standard_tissue_means(args.save_standard_tissue_means)

    def save_standard_tissue_means(self, filename: intnormt.PathLike, /) -> None:
        if self.standard_tissue_means is None:
            msg = "Fit required before saving standard tissue means."
            raise RuntimeError(msg)
        np.save(filename, self.standard_tissue_means)

    def load_standard_tissue_means(self, filename: intnormt.PathLike, /) -> None:
        data = np.load(filename)
        self.standard_tissue_means = data

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace, /) -> LeastSquaresNormalize:
        out = cls(norm_value=args.norm_value)
        return out

    def call_from_argparse_args(self, args: argparse.Namespace, /) -> None:
        if args.load_standard_tissue_means is not None:
            self.load_standard_tissue_means(args.load_standard_tissue_means)
            self.fit = lambda *args, **kwargs: None  # type: ignore[assignment]

        args.modality = intnormt.Modalities.from_string(args.modality)
        if args.mask_dir is not None:
            if args.modality != intnormt.Modalities.T1:
                msg = f"If brain masks provided, modality must be 't1'. Got '{args.modality}'."  # noqa: E501
                raise ValueError(msg)
        elif args.tissue_membership_dir is not None:
            args.mask_dir = args.tissue_membership_dir
        super().call_from_argparse_args(args)

    @classmethod
    def get_parent_parser(
        cls,
        desc: builtins.str,
        valid_modalities: builtins.frozenset[builtins.str] = intnorm.VALID_MODALITIES,
        **kwargs: typing.Any,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "image_dir",
            type=intnormt.dir_path(),
            help="Path of directory containing images to normalize.",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=intnormt.dir_path(),
            default=None,
            help="Path of directory in which to save normalized images.",
        )
        parser.add_argument(
            "-mo",
            "--modality",
            type=str,
            default="t1",
            choices=intnorm.VALID_MODALITIES,
            help="Modality of the images.",
        )
        parser.add_argument(
            "-n",
            "--norm-value",
            type=intnormt.positive_float(),
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
            "-sstm",
            "--save-standard-tissue-means",
            default=None,
            type=intnormt.save_file_path(),
            help="Save the standard tissue means fit by the method.",
        )
        parser.add_argument(
            "-lstm",
            "--load-standard-tissue-means",
            default=None,
            type=intnormt.file_path(),
            help="Load a standard tissue means previously fit by the method.",
        )
        exclusive = parent_parser.add_argument_group(
            "mutually exclusive optional arguments"
        )
        group = exclusive.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "-m",
            "--mask-dir",
            type=intnormt.dir_path(),
            default=None,
            help="Path to a foreground mask for the image. "
            "Provide this if not providing a tissue mask "
            "(if image is not skull-stripped).",
        )
        group.add_argument(
            "-tm",
            "--tissue-membership-dir",
            type=intnormt.dir_path(),
            help="Path to a mask of a tissue memberships. "
            "Provide this if not providing the foreground mask.",
        )
        return parent_parser
