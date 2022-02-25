"""WhiteStripe (normal-appearing white matter mean & standard deviation) normalization
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""

from __future__ import annotations

__all__ = ["WhiteStripeNormalize"]

import argparse
import builtins
import typing
import warnings

import numpy as np
import numpy.typing as npt

import intensity_normalization.normalize.base as intnormb
import intensity_normalization.typing as intnormt
import intensity_normalization.util.histogram_tools as intnormhisttool


class WhiteStripeNormalize(
    intnormb.LocationScaleCLIMixin, intnormb.SingleImageNormalizeCLI
):
    """
    find the normal appearing white matter of the input MR image and
    use those values to standardize the data (i.e., subtract the mean of
    the values in the indices and divide by the std of those values)
    """

    def __init__(
        self,
        *,
        norm_value: builtins.float = 1.0,
        width: builtins.float = 0.05,
        width_l: builtins.float | None = None,
        width_u: builtins.float | None = None,
        **kwargs: typing.Any,
    ):
        super().__init__(norm_value=norm_value, **kwargs)
        if norm_value != 1.0:
            warnings.warn("norm_value not used in RavelNormalize")
        self.width_l = width_l or width
        self.width_u = width_u or width
        self.whitestripe: npt.NDArray | None = None

    def calculate_location(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> builtins.float:
        loc: builtins.float = image[self.whitestripe].mean()
        return loc

    def calculate_scale(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> builtins.float:
        scale: builtins.float = image[self.whitestripe].std()
        return scale

    def setup(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
    ) -> None:
        if modality is None:
            modality = "t1"
        mask = self._get_mask(image, mask, modality=modality)
        masked = image * mask
        voi = image[mask]
        wm_mode = intnormhisttool.get_tissue_mode(voi, modality=modality)
        wm_mode_quantile: builtins.float = np.mean(voi < wm_mode).item()
        lower_bound = max(wm_mode_quantile - self.width_l, 0.0)
        upper_bound = min(wm_mode_quantile + self.width_u, 1.0)
        ws_l: builtins.float
        ws_u: builtins.float
        ws_l, ws_u = np.quantile(voi, (lower_bound, upper_bound))  # type: ignore[misc]
        self.whitestripe = (masked > ws_l) & (masked < ws_u)

    def teardown(self) -> None:
        del self.whitestripe

    @staticmethod
    def name() -> builtins.str:
        return "ws"

    @staticmethod
    def fullname() -> builtins.str:
        return "WhiteStripe"

    @staticmethod
    def description() -> builtins.str:
        return "Standardize the normal appearing WM of a MR image."

    @staticmethod
    def add_method_specific_arguments(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("method-specific arguments")
        parser.add_argument(
            "--width",
            default=0.05,
            type=float,
            help="Width of the 'white stripe'. (See original paper.)",
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace, /) -> WhiteStripeNormalize:
        return cls(width=args.width)

    def plot_histogram_from_args(
        self,
        args: argparse.Namespace,
        /,
        normalized: intnormt.ImageLike,
        mask: intnormt.ImageLike | None = None,
    ) -> None:
        if mask is None:
            mask = self.estimate_foreground(normalized)
        super().plot_histogram_from_args(args, normalized, mask)
