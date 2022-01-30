"""Base class for normalization methods
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 01 Jun 2021
"""
# For MRO information use help(...) on a class and refer to the following
# https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

from __future__ import annotations

__all__ = [
    "SingleImageNormalizeCLI",
    "DirectoryNormalizeCLI",
    "LocationScaleCLIMixin",
]

import abc
import argparse
import builtins
import logging
import pathlib
import typing
import warnings

import pymedio.image as mioi

import intensity_normalization.base_cli as intnormcli
import intensity_normalization.plot.histogram as intnormhist
import intensity_normalization.typing as intnormt
import intensity_normalization.util.io as intnormio

logger = logging.getLogger(__name__)

T = typing.TypeVar("T")


class NormalizeMixin(metaclass=abc.ABCMeta):
    def __call__(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
        **kwargs: typing.Any,
    ) -> intnormt.Image:
        return self.normalize_image(image, mask, modality=modality)

    @abc.abstractmethod
    def normalize_image(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
    ) -> intnormt.Image:
        raise NotImplementedError

    def setup(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
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
        image: intnormt.Image, /, *, background_threshold: builtins.float = 1e-6
    ) -> intnormt.Image:
        if image.min() < 0.0:
            msg = "Data contains negative values; "
            msg += "skull-stripped functionality assumes "
            msg += "the foreground is all positive. "
            msg += "Provide the brain mask if otherwise."
            warnings.warn(msg)
        ss_foreground: intnormt.Image = image > background_threshold
        return ss_foreground

    def _get_mask(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
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
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
    ) -> intnormt.Image:
        voi: intnormt.Image = image[self._get_mask(image, mask, modality=modality)]
        return voi


class LocationScaleMixin(NormalizeMixin, metaclass=abc.ABCMeta):
    def __init__(self, *, norm_value: builtins.float = 1.0, **kwargs: typing.Any):
        super().__init__(**kwargs)
        self.norm_value = norm_value

    @abc.abstractmethod
    def calculate_location(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
    ) -> builtins.float:
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_scale(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
    ) -> builtins.float:
        raise NotImplementedError

    def normalize_image(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
    ) -> intnormt.Image:
        self.setup(image, mask, modality=modality)
        loc = self.calculate_location(image, mask, modality=modality)
        scale = self.calculate_scale(image, mask, modality=modality)
        self.teardown()
        normalized: intnormt.Image = ((image - loc) / scale) * self.norm_value
        return normalized


class NormalizeCLIMixin(NormalizeMixin, intnormcli.CLIMixin, metaclass=abc.ABCMeta):
    def normalize_from_filename(
        self,
        image_path: intnormt.PathLike,
        /,
        mask_path: intnormt.PathLike | None = None,
        *,
        out_path: intnormt.PathLike | None = None,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
    ) -> intnormt.Image:
        image = mioi.Image.from_path(image_path)
        mask = mioi.Image.from_path(mask_path) if mask_path is not None else None
        if out_path is None:
            out_path = self.append_name_to_file(image_path)
        logger.info(f"Normalizing image: {image_path}")
        normalized: mioi.Image = self.normalize_image(image, mask, modality=modality)
        logger.info(f"Saving normalized image: {out_path}")
        normalized.save(out_path, squeeze=False)
        return normalized, mask

    @abc.abstractmethod
    def plot_histogram_from_args(
        self,
        args: argparse.Namespace,
        /,
        normalized: intnormt.Image,
        mask: intnormt.Image | None = None,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def get_parent_parser(
        cls,
        desc: builtins.str,
        **kwargs,
    ) -> argparse.ArgumentParser:
        parser = super().get_parent_parser(desc, **kwargs)
        parser.add_argument(
            "-p",
            "--plot-histogram",
            action="store_true",
            help="Plot the histogram of the normalized image.",
        )
        return parser

    @abc.abstractmethod
    def call_from_argparse_args(self, args: argparse.Namespace, /) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_argparse_args(cls: typing.Type[T], args: argparse.Namespace, /) -> T:
        raise NotImplementedError

    def save_additional_info(
        self,
        args: argparse.Namespace,
        **kwargs: typing.Any,
    ) -> None:
        return


class LocationScaleCLIMixin(LocationScaleMixin, NormalizeCLIMixin):
    @classmethod
    def get_parent_parser(
        cls,
        desc: builtins.str,
        **kwargs,
    ) -> argparse.ArgumentParser:
        parser = super().get_parent_parser(desc, **kwargs)
        parser.add_argument(
            "-n",
            "--norm-value",
            type=intnormt.positive_float(),
            default=1.0,
            help="Reference value for normalization.",
        )
        return parser

    @classmethod
    def from_argparse_args(cls: typing.Type[T], args: argparse.Namespace, /) -> T:
        return cls(norm_value=args.norm_value)


class SingleImageNormalizeCLI(NormalizeCLIMixin, intnormcli.SingleImageCLI):
    def plot_histogram_from_args(
        self,
        args: argparse.Namespace,
        /,
        normalized: intnormt.Image,
        mask: intnormt.Image | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if args.output is None:
            output = pathlib.Path(args.image).parent / "hist.pdf"
        else:
            output = pathlib.Path(args.output).parent / "hist.pdf"
        ax = intnormhist.plot_histogram(normalized, mask)
        ax.set_title(self.fullname())
        plt.savefig(output)

    def call_from_argparse_args(self, args: argparse.Namespace, /) -> None:
        normalized, mask = self.normalize_from_filename(
            args.image,
            args.mask,
            out_path=args.output,
            modality=intnormt.Modalities.from_string(args.modality),
        )
        if args.plot_histogram:
            self.plot_histogram_from_args(args, normalized, mask)
        self.save_additional_info(args, normalized=normalized, mask=mask)


class SampleNormalizeCLIMixin(NormalizeCLIMixin, intnormcli.CLIMixin):
    def fit(
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
        **kwargs: typing.Any,
    ) -> None:
        return None

    def process_directories(
        self,
        image_dir: intnormt.PathLike,
        /,
        mask_dir: intnormt.PathLike | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
        ext: builtins.str = "nii*",
        return_normalized_and_masks: builtins.bool = False,
        **kwargs: typing.Any,
    ) -> typing.Tuple[
        typing.Sequence[intnormt.Image], typing.Sequence[intnormt.Image | None]
    ] | None:
        logger.debug("Grabbing images")
        images, masks = intnormio.gather_images_and_masks(image_dir, mask_dir, ext=ext)
        self.fit(images, masks, modality=modality, **kwargs)
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
        normalized: typing.Sequence[intnormt.Image],
        masks: typing.Sequence[intnormt.Image | None] | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if args.output_dir is None:
            output = pathlib.Path(args.image_dir) / "hist.pdf"
        else:
            output = pathlib.Path(args.output_dir) / "hist.pdf"
        hp = intnormhist.HistogramPlotter(title=self.fullname())
        _ = hp(normalized, masks)
        plt.savefig(output)

    def call_from_argparse_args(self, args: argparse.Namespace, /) -> None:
        out = self.process_directories(
            args.image_dir,
            args.mask_dir,
            modality=intnormt.Modalities.from_string(args.modality),  # type: ignore[attr-defined]
            ext=args.extension,
            return_normalized_and_masks=True,
        )
        assert out is not None
        normalized, masks = out
        assert isinstance(normalized, list)
        image_filenames = intnormio.glob_ext(args.image_dir, ext=args.extension)
        output_filenames = [
            self.append_name_to_file(fn, args.output_dir) for fn in image_filenames
        ]
        n_images = len(normalized)
        assert n_images == len(output_filenames)
        for i, (norm_image, fn) in enumerate(zip(normalized, output_filenames), 1):
            logger.info(f"Saving normalized image: {fn} ({i}/{n_images})")
            norm_image.view(mioi.Image).save(fn)
        self.save_additional_info(
            args,
            normalized=normalized,
            masks=masks,
            image_filenames=image_filenames,
        )
        if args.plot_histogram:
            self.plot_histogram_from_args(args, normalized, masks)

    @classmethod
    def from_argparse_args(cls: typing.Type[T], args: argparse.Namespace, /) -> T:
        return cls()  # TODO: why is this empty?


class DirectoryNormalizeCLI(
    SampleNormalizeCLIMixin, intnormcli.DirectoryCLI, metaclass=abc.ABCMeta
):
    def fit(
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
        **kwargs: typing.Any,
    ) -> None:
        images, masks = self.before_fit(images, masks, modality=modality, **kwargs)
        logger.info("Fitting")
        self._fit(images, masks, modality=modality, **kwargs)
        logger.debug("Done fitting")

    def _fit(
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
        **kwargs: typing.Any,
    ) -> None:
        raise NotImplementedError

    def before_fit(
        self,
        images: typing.Sequence[intnormt.Image],
        /,
        masks: typing.Sequence[intnormt.Image] | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
        **kwargs: typing.Any,
    ) -> typing.Tuple[
        typing.Sequence[intnormt.Image], typing.Sequence[intnormt.Image] | None
    ]:
        assert len(images) > 0
        logger.info("Loading data")
        if hasattr(images[0], "get_fdata"):
            images = [img.get_fdata() for img in images]  # type: ignore[attr-defined]
        if masks is not None:
            if hasattr(masks[0], "get_fdata"):
                masks = [msk.get_fdata() for msk in masks]  # type: ignore[attr-defined]
        logger.debug("Loaded data")
        return images, masks

    def fit_from_directories(
        self,
        image_dir: intnormt.PathLike,
        /,
        mask_dir: intnormt.PathLike | None = None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,  # type: ignore[attr-defined]
        ext: builtins.str = "nii*",
        return_normalized_and_masks: builtins.bool = False,
        **kwargs: typing.Any,
    ) -> typing.Tuple[
        typing.Sequence[intnormt.Image], typing.Sequence[intnormt.Image | None]
    ] | None:
        return self.process_directories(
            image_dir,
            mask_dir,
            modality=modality,
            ext=ext,
            return_normalized_and_masks=return_normalized_and_masks,
            **kwargs,
        )
