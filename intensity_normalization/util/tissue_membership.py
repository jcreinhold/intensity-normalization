"""Find the tissue-membership of a T1-w brain image
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["find_tissue_memberships", "TissueMembershipFinder"]

import argparse
import builtins
import operator

import nibabel as nib
import numpy as np
from skfuzzy import cmeans

import intensity_normalization.base_cli as intnormcli
import intensity_normalization.typing as intnormt


def find_tissue_memberships(
    image: intnormt.Image,
    /,
    mask: intnormt.Image | None = None,
    *,
    hard_segmentation: builtins.bool = False,
    n_classes: builtins.int = 3,
) -> intnormt.Image:
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
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive. Got {n_classes}")
    if mask is None:
        mask = image > 0.0
    else:
        mask = mask > 0.0
    foreground_size = mask.sum()
    foreground = image[mask].reshape(-1, foreground_size)
    centers, memberships_, *_ = cmeans(foreground, n_classes, 2, 0.005, 50)
    # sort the tissue memberships to CSF/GM/WM (assuming T1-w image)
    sorted_memberships = sorted(zip(centers, memberships_), key=operator.itemgetter(0))
    memberships = [m for _, m in sorted_memberships]
    tissue_mask = np.zeros(image.shape + (n_classes,))
    for i in range(n_classes):
        tissue_mask[..., i][mask] = memberships[i]
    if hard_segmentation:
        tmp_mask = np.zeros(image.shape)
        masked = tissue_mask[mask]
        tmp_mask[mask] = np.argmax(masked, axis=1) + 1
        tissue_mask = tmp_mask
    return tissue_mask


class TissueMembershipFinder(intnormcli.CLI):
    def __init__(self, hard_segmentation: builtins.bool = False):
        self.hard_segmentation = hard_segmentation

    def __call__(
        self, image: intnormt.Image, /, mask: intnormt.Image | None = None, **kwargs
    ) -> intnormt.Image:
        tissue_memberships = find_tissue_memberships(
            image,
            mask,
            hard_segmentation=self.hard_segmentation,
        )
        out = nib.Nifti1Image(tissue_memberships, image.affine)
        return out

    @staticmethod
    def name() -> builtins.str:
        return "tm"

    @staticmethod
    def fullname() -> builtins.str:
        return "tissue_membership"

    @staticmethod
    def description() -> builtins.str:
        return "Find tissue memberships of an MR image."

    @staticmethod
    def get_parent_parser(desc: builtins.str, **kwargs) -> argparse.ArgumentParser:
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
            type=intnormt.save_nifti_path(),
            default=None,
            help="Path to save registered image.",
        )
        parser.add_argument(
            "-hs",
            "--hard-segmentation",
            action="store_true",
            help="classify tissue memberships into segmentation",
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
            help="print the version of intensity-normalization",
        )
        return parser

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace) -> TissueMembershipFinder:
        return cls(args.hard_segmentation)
