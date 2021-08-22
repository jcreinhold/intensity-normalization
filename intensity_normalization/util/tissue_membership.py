# -*- coding: utf-8 -*-
"""
intensity_normalization.util.tissue_membership

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "find_tissue_memberships",
    "TissueMembershipFinder",
]

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Optional, Tuple, Type, TypeVar

import nibabel as nib
import numpy as np
from skfuzzy import cmeans

from intensity_normalization.parse import CLIParser
from intensity_normalization.type import Array, NiftiImage, file_path, save_nifti_path


def find_tissue_memberships(
    image: Array,
    mask: Array = None,
    hard_segmentation: bool = False,
) -> Array:
    """Tissue memberships for a T1-w brain image with fuzzy c-means

    Args:
        image: image to find tissue masks for (must be T1-w)
        mask: mask covering the brain of image (none if already skull-stripped)
        hard_segmentation: pick the maximum membership as the true class in output

    Returns:
        tissue_mask: membership values for each of three classes in the image
            (or class determinations w/ hard_seg)
    """
    if mask is None:
        mask = image > 0.0
    else:
        mask = mask > 0.0
    assert isinstance(mask, Array)
    foreground_size = mask.sum()
    foreground = image[mask].reshape(-1, foreground_size)
    centers, memberships_, *_ = cmeans(foreground, 3, 2, 0.005, 50)

    def get_center(element: Tuple[float, Array]) -> float:
        center: float = element[0]
        return center

    # sort the tissue memberships to CSF/GM/WM (assuming T1-w image)
    sorted_memberships = sorted(zip(centers, memberships_), key=get_center)
    memberships = [m for _, m in sorted_memberships]
    tissue_mask = np.zeros(image.shape + (3,))
    for i in range(3):
        tissue_mask[..., i][mask] = memberships[i]
    if hard_segmentation:
        tmp_mask = np.zeros(image.shape)
        masked = tissue_mask[mask]
        tmp_mask[mask] = np.argmax(masked, axis=1) + 1
        tissue_mask = tmp_mask
    return tissue_mask


TMF = TypeVar("TMF", bound="TissueMembershipFinder")


class TissueMembershipFinder(CLIParser):
    def __init__(self, hard_segmentation: bool = False):
        self.hard_segmentation = hard_segmentation

    def __call__(  # type: ignore[override]
        self,
        image: NiftiImage,
        mask: Optional[NiftiImage] = None,
    ) -> NiftiImage:
        data = image.get_fdata()
        mask = mask and mask.get_fdata()
        tissue_memberships = find_tissue_memberships(
            data,
            mask,
            self.hard_segmentation,
        )
        out = nib.Nifti1Image(tissue_memberships, image.affine)
        return out

    def name(self) -> str:  # type: ignore[override]
        base = "tissue_"
        suffix = "mask" if self.hard_segmentation else "membership"
        return base + suffix

    @staticmethod
    def description() -> str:
        return "Find tissue memberships of an MR image."

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
            "-m",
            "--mask",
            type=file_path(),
            default=None,
            help="Path of foreground mask for image.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=save_nifti_path(),
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
        return parser

    @classmethod
    def from_argparse_args(cls: Type[TMF], args: Namespace) -> TMF:
        return cls(args.hard_segmentation)
