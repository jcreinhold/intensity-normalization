"""Base class for normalization methods
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: Jun 01, 2021
"""

from __future__ import annotations

__all__ = [
    "NormalizeBase",
    "NormalizeFitBase",
]

import argparse
import builtins
import logging
import pathlib
import typing
import warnings

import nibabel as nib

import intensity_normalization as intnorm
import intensity_normalization.base_cli as intnormcli
import intensity_normalization.plot.histogram as intnormhist
import intensity_normalization.typing as intnormt
import intensity_normalization.util.io as intnormio

logger = logging.getLogger(__name__)


class NormalizeBase(intnormcli.CLI):
    def __init__(self, norm_value: builtins.float = 1.0):
        self.norm_value = norm_value

    def __call__(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
    ) -> intnormt.Image:
        return self.normalize_image(image, mask, modality=modality)

    def normalize_image(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
    ) -> intnormt.Image:
        self.setup(image, mask, modality=modality)
        loc = self.calculate_location(image, mask, modality=modality)
        scale = self.calculate_scale(image, mask, modality=modality)
        self.teardown()
        normalized: intnormt.Image = ((image - loc) / scale) * self.norm_value
        return normalized

    def normalize_from_filename(
        self,
        image_path: intnormt.PathLike,
        /,
        mask_path: intnormt.PathLike | None = None,
        out_path: intnormt.PathLike | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
    ) -> intnormt.Image:
        image = nib.load(image_path)
        mask = nib.load(mask_path) if mask_path is not None else None
        if out_path is None:
            out_path = self.append_name_to_file(image_path)
        logger.info(f"Normalizing image: {image_path}")
        normalized = self.normalize_image(image, mask, modality=modality)
        logger.info(f"Saving normalized image: {out_path}")
        normalized.to_filename(out_path)
        return normalized, mask

    def plot_histogram_from_args(
        self,
        args: argparse.Namespace,
        /,
        normalized: intnormt.Image,
        mask: intnormt.Image | None = None,
    ) -> None:
        from pathlib import Path

        import matplotlib.pyplot as plt

        if args.output is None:
            output = Path(args.image).parent / "hist.pdf"
        else:
            output = Path(args.output).parent / "hist.pdf"
        normalized_data = normalized.get_fdata()
        mask_data = mask and mask.get_fdata()
        ax = intnormhist.plot_histogram(normalized_data, mask_data)
        ax.set_title(self.fullname())
        plt.savefig(output)

    def call_from_argparse_args(self, args: argparse.Namespace) -> None:
        normalized, mask = self.normalize_from_filename(
            args.image,
            args.mask,
            args.output,
            modality=args.modality,
        )
        if args.plot_histogram:
            self.plot_histogram_from_args(args, normalized, mask)
        self.save_additional_info(args, normalized=normalized, mask=mask)

    def calculate_location(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
    ) -> float:
        raise NotImplementedError

    def calculate_scale(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
    ) -> float:
        raise NotImplementedError

    def setup(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
    ) -> None:
        return

    def teardown(self) -> None:
        return

    @staticmethod
    def estimate_foreground(image: intnormt.Image, /) -> intnormt.Image:
        foreground: intnormt.Image = image > image.mean()
        return foreground

    @staticmethod
    def skull_stripped_foreground(
        data: intnormt.Image, /, *, background_threshold: builtins.float = 1e-6
    ) -> intnormt.Image:
        if data.min() < 0.0:
            msg = (
                "Data contains negative values; "
                "skull-stripped functionality assumes "
                "the foreground is all positive. "
                "Provide the brain mask if otherwise."
            )
            warnings.warn(msg)
        ss_foreground: intnormt.Image = data > background_threshold
        return ss_foreground

    def _get_mask(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
        background_threshold: builtins.float = 1e-6,
    ) -> intnormt.Image:
        if mask is None:
            mask = self.skull_stripped_foreground(
                image, background_threshold=background_threshold
            )
        out: intnormt.Image = mask > 0.0
        return out

    def _get_voi(
        self,
        data: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
    ) -> intnormt.Image:
        voi: intnormt.Image = data[self._get_mask(data, mask, modality=modality)]
        return voi

    @staticmethod
    def get_parent_parser(
        desc: builtins.str,
        valid_modalities: typing.Set[builtins.str] = intnorm.VALID_MODALITIES,
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
            type=intnormt.save_nifti_path(),
            default=None,
            help="Path to save normalized image.",
        )
        parser.add_argument(
            "-mo",
            "--modality",
            type=str,
            default=None,
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
            help="print the version of intensity-normalization",
        )
        return parser

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace) -> NormalizeBase:
        return cls(args.norm_value)

    def save_additional_info(
        self,
        args: argparse.Namespace,
        **kwargs,
    ) -> None:
        return


class NormalizeSampleBase(NormalizeBase):
    def fit(self, *args, **kwargs) -> None:
        return None

    def process_directories(
        self,
        image_dir: intnormt.PathLike,
        /,
        mask_dir: intnormt.PathLike | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
        ext: builtins.str = "nii*",
        return_normalized_and_masks: builtins.bool = False,
        **kwargs,
    ):
        logger.debug("Grabbing images")
        images, masks = intnormio.gather_images_and_masks(image_dir, mask_dir, ext)
        self.fit(images, masks, modality, **kwargs)
        if return_normalized_and_masks:
            normalized: typing.List[intnormt.Image] = []
            n_images = len(images)
            assert n_images == len(masks)
            for i, (image, mask) in enumerate(zip(images, masks), 1):
                logger.info(f"Normalizing image {i}/{n_images}")
                normalized.append(self(image, mask, modality=modality))
            return normalized, masks
        return None

    def plot_histogram_from_args(
        self,
        args: argparse.Namespace,
        /,
        normalized: intnormt.Image,
        masks: intnormt.Image | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if args.output_dir is None:
            output = pathlib.Path(args.image_dir) / "hist.pdf"
        else:
            output = pathlib.Path(args.output_dir) / "hist.pdf"
        hp = intnormhist.HistogramPlotter(title=self.fullname())
        _ = hp(normalized, masks)
        plt.savefig(output)

    def call_from_argparse_args(self, args: argparse.Namespace) -> None:
        normalized, masks = self.process_directories(
            args.image_dir,
            args.mask_dir,
            modality=args.modality,
            ext=args.extension,
            return_normalized_and_masks=True,
        )
        assert isinstance(normalized, list)
        image_filenames = intnormio.glob_ext(args.image_dir)
        output_filenames = [
            self.append_name_to_file(fn, args.output_dir) for fn in image_filenames
        ]
        n_images = len(normalized)
        assert n_images == len(output_filenames)
        for i, (norm_image, fn) in enumerate(zip(normalized, output_filenames), 1):
            logger.info(f"Saving normalized image: {fn} ({i}/{n_images})")
            norm_image.to_filename(fn)
        self.save_additional_info(
            args,
            normalized=normalized,
            masks=masks,
            image_filenames=image_filenames,
        )
        if args.plot_histogram:
            self.plot_histogram_from_args(args, normalized, masks)

    @staticmethod
    def get_parent_parser(
        desc: builtins.str,
        valid_modalities: typing.Set[builtins.str] = intnorm.VALID_MODALITIES,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "image_dir",
            type=intnormt.dir_path(),
            help="Path of directory of images to normalize.",
        )
        parser.add_argument(
            "-m",
            "--mask-dir",
            type=intnormt.dir_path(),
            default=None,
            help="Path of directory of foreground masks corresponding to images.",
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
            default=None,
            choices=intnorm.VALID_MODALITIES,
            help="Modality of the images.",
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
            help="print the version of intensity-normalization",
        )
        return parser

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace) -> NormalizeSampleBase:
        out = cls()
        return out


class NormalizeFitBase(NormalizeSampleBase):
    def fit(
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
        **kwargs,
    ) -> None:
        images, masks = self.before_fit(images, masks, modality=modality, **kwargs)
        logger.info("Fitting")
        self._fit(images, masks, modality=modality, **kwargs)
        logger.debug("Done fitting")

    def _fit(  # type: ignore[no-untyped-def]
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def before_fit(
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
        **kwargs,
    ) -> typing.Tuple[
        typing.Sequence[intnormt.Image], typing.Sequence[intnormt.Image | None]
    ]:
        assert len(images) > 0
        logger.info("Loading data")
        if hasattr(images[0], "get_fdata"):
            images = [img.get_fdata() for img in images]
        if masks is not None:
            if hasattr(masks[0], "get_fdata"):
                masks = [msk.get_fdata() for msk in masks]
        logger.debug("Loaded data")
        return images, masks

    def fit_from_directories(
        self,
        image_dir: intnormt.PathLike,
        /,
        mask_dir: intnormt.PathLike | None = None,
        *,
        modality: intnorm.Modalities = intnorm.Modalities.T1,
        ext: builtins.str = "nii*",
        return_normalized_and_masks: builtins.bool = False,
        **kwargs,
    ) -> typing.Tuple[
        typing.Sequence[intnormt.Image], typing.Sequence[intnormt.Image | None]
    ] | None:
        return self.process_directories(
            image_dir=image_dir,
            mask_dir=mask_dir,
            modality=modality,
            ext=ext,
            return_normalized_and_masks=return_normalized_and_masks,
            **kwargs,
        )
