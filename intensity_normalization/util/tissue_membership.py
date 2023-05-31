"""Find the tissue-membership of a T1-w brain image
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["find_tissue_memberships", "TissueMembershipFinder"]

import argparse
import operator
import typing

import numpy as np
import numpy.typing as npt
import pymedio.image as mioi
from skfuzzy import cmeans

import intensity_normalization as intnorm
import intensity_normalization.base_cli as intnormcli
import intensity_normalization.typing as intnormt


def find_tissue_memberships(
    image: intnormt.ImageLike,
    /,
    mask: intnormt.ImageLike | None = None,
    *,
    hard_segmentation: bool = False,
    n_classes: int = 3,
) -> mioi.Image:
    """Tissue memberships for a T1-w brain image with fuzzy c-means

    Args:
        image: image to find tissue masks for (must be T1-w)
        mask: mask covering the brain of image (none if already skull-stripped)
        hard_segmentation: pick the maximum membership as the true class in output
        n_classes: number of classes (usually three for CSF, GM, WM)

    Returns:
        tissue_mask: membership values for each of three classes in the image
            (or class determinations w/ hard_seg)
    """
    _image: np.ndarray = np.array(image, copy=True)  # some op mutates original o/w
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive. Got '{n_classes}'.")
    if mask is None:
        mask = _image > 0.0
    else:
        mask = mask > 0.0
    assert mask is not None
    foreground_size = typing.cast(int, mask.sum())
    foreground = _image[mask].reshape(-1, foreground_size)
    centers, memberships_, *_ = cmeans(foreground, n_classes, 2, 0.005, 50)
    # sort the tissue memberships to CSF/GM/WM (assuming T1-w image)
    sorted_memberships = sorted(zip(centers, memberships_), key=operator.itemgetter(0))
    memberships = [m for _, m in sorted_memberships]
    tissue_mask = np.zeros(_image.shape + (n_classes,))
    for i in range(n_classes):
        tissue_mask[..., i][mask] = memberships[i]
    if hard_segmentation:
        tmp_mask = np.zeros(_image.shape)
        masked = tissue_mask[mask]
        tmp_mask[mask] = np.argmax(masked, axis=1) + 1
        tissue_mask = tmp_mask
    affine: npt.NDArray | None
    if hasattr(image, "affine"):
        affine = image.affine.copy()
    else:
        affine = None
    return mioi.Image(tissue_mask, affine=affine)


class TissueMembershipFinder(intnormcli.SingleImageCLI):
    def __init__(self, hard_segmentation: bool = False):
        super().__init__()
        self.hard_segmentation = hard_segmentation

    def __call__(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        **kwargs: typing.Any,
    ) -> intnormt.ImageLike:
        tissue_memberships = find_tissue_memberships(
            image,
            mask,
            hard_segmentation=self.hard_segmentation,
        )
        return tissue_memberships

    @staticmethod
    def name() -> str:
        return "tm"

    @staticmethod
    def fullname() -> str:
        return "tissue_membership"

    @staticmethod
    def description() -> str:
        return "Find tissue memberships of an MR image."

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
            "-m",
            "--mask",
            type=intnormt.file_path(),
            default=None,
            help="Path of foreground mask for image.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=intnormt.save_file_path(),
            default=None,
            help="Path to save preprocessed image.",
        )
        parser.add_argument(
            "-hs",
            "--hard-segmentation",
            action="store_true",
            help="Classify tissue memberships into segmentation",
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

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace) -> TissueMembershipFinder:
        return cls(args.hard_segmentation)
