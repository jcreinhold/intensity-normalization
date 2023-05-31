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
import collections.abc
import logging
import pathlib
import typing
import warnings

import pymedio.image as mioi

import intensity_normalization as intnorm
import intensity_normalization.base_cli as intnormcli
import intensity_normalization.typing as intnormt
import intensity_normalization.util.io as intnormio

logger = logging.getLogger(__name__)

T = typing.TypeVar("T")
ImageSeq = collections.abc.Sequence[intnormt.ImageLike]
MaskSeqOrNone = typing.Union[ImageSeq, None]


class NormalizeMixin(metaclass=abc.ABCMeta):
    def __call__(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        **kwargs: typing.Any,
    ) -> intnormt.ImageLike:
        return self.normalize_image(image, mask, modality=modality)

    @abc.abstractmethod
    def normalize_image(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> intnormt.ImageLike:
        raise NotImplementedError

    def setup(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> None:
        return

    def teardown(self) -> None:
        return

    @staticmethod
    def estimate_foreground(image: intnormt.ImageLike, /) -> intnormt.ImageLike:
        foreground: intnormt.ImageLike = image > image.mean()
        return foreground

    @staticmethod
    def skull_stripped_foreground(
        image: intnormt.ImageLike, /, *, background_threshold: float = 1e-6
    ) -> intnormt.ImageLike:
        if image.min() < 0.0:
            msg = "Data contains negative values; "
            msg += "skull-stripped functionality assumes "
            msg += "the foreground is all positive. "
            msg += "Provide the brain mask if otherwise."
            warnings.warn(msg)
        ss_foreground: intnormt.ImageLike = image > background_threshold
        return ss_foreground

    def _get_mask(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        background_threshold: float = 1e-6,
    ) -> intnormt.ImageLike:
        if mask is None:
            mask = self.skull_stripped_foreground(
                image, background_threshold=background_threshold
            )
        out: intnormt.ImageLike = mask > 0.0
        return out

    def _get_voi(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> intnormt.ImageLike:
        voi: intnormt.ImageLike = image[self._get_mask(image, mask, modality=modality)]
        return voi


class LocationScaleMixin(NormalizeMixin, metaclass=abc.ABCMeta):
    def __init__(self, *, norm_value: float = 1.0, **kwargs: typing.Any):
        super().__init__(**kwargs)
        self.norm_value = norm_value

    @abc.abstractmethod
    def calculate_location(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_scale(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> float:
        raise NotImplementedError

    def normalize_image(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> intnormt.ImageLike:
        self.setup(image, mask, modality=modality)
        loc = self.calculate_location(image, mask, modality=modality)
        scale = self.calculate_scale(image, mask, modality=modality)
        self.teardown()
        normalized: intnormt.ImageLike = (image - loc) * (self.norm_value / scale)
        return normalized


class NormalizeCLIMixin(NormalizeMixin, intnormcli.CLIMixin, metaclass=abc.ABCMeta):
    def normalize_from_filename(
        self,
        image_path: intnormt.PathLike,
        /,
        mask_path: intnormt.PathLike | None = None,
        *,
        out_path: intnormt.PathLike | None = None,
        modality: intnormt.Modality = intnormt.Modality.T1,
    ) -> tuple[mioi.Image, mioi.Image | None]:
        image: mioi.Image = mioi.Image.from_path(image_path)
        mask: typing.Optional[mioi.Image]
        mask = None if mask_path is None else mioi.Image.from_path(mask_path)
        if out_path is None:
            out_path = self.append_name_to_file(image_path)
        logger.info(f"Normalizing image: {image_path}")
        normalized = typing.cast(
            mioi.Image, self.normalize_image(image, mask, modality=modality)
        )
        logger.info(f"Saving normalized image: {out_path}")
        normalized.to_filename(out_path)
        return normalized, mask

    @classmethod
    def get_parent_parser(
        cls,
        desc: str,
        valid_modalities: frozenset[str] = intnorm.VALID_MODALITIES,
        **kwargs: typing.Any,
    ) -> argparse.ArgumentParser:
        parser = super().get_parent_parser(
            desc, valid_modalities=valid_modalities, **kwargs
        )
        parser.add_argument(
            "-p",
            "--plot-histogram",
            action="store_true",
            help="Plot the histogram of the normalized image.",
        )
        return parser

    @abc.abstractmethod
    def call_from_argparse_args(
        self, args: argparse.Namespace, /, **kwargs: typing.Any
    ) -> None:
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
        desc: str,
        valid_modalities: frozenset[str] = intnorm.VALID_MODALITIES,
        **kwargs: typing.Any,
    ) -> argparse.ArgumentParser:
        parser = super().get_parent_parser(
            desc, valid_modalities=valid_modalities, **kwargs
        )
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
        return cls(norm_value=args.norm_value)  # type: ignore[call-arg]


class SingleImageNormalizeCLI(NormalizeCLIMixin, intnormcli.SingleImageCLI):
    def plot_histogram_from_args(
        self,
        args: argparse.Namespace,
        /,
        normalized: intnormt.ImageLike,
        mask: intnormt.ImageLike | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        import intensity_normalization.plot.histogram as intnormhist

        if args.output is None:
            output = pathlib.Path(args.image).parent / "hist.pdf"
        else:
            output = pathlib.Path(args.output).parent / "hist.pdf"
        ax = intnormhist.plot_histogram(normalized, mask)
        ax.set_title(self.fullname())
        plt.savefig(output)

    def call_from_argparse_args(
        self, args: argparse.Namespace, /, **kwargs: typing.Any
    ) -> None:
        normalized, mask = self.normalize_from_filename(
            args.image,
            args.mask,
            out_path=args.output,
            modality=intnormt.Modality.from_string(args.modality),
        )
        if args.plot_histogram:
            self.plot_histogram_from_args(args, normalized, mask)
        self.save_additional_info(args, normalized=normalized, mask=mask)


class SampleNormalizeCLIMixin(NormalizeCLIMixin, intnormcli.CLIMixin):
    def fit(
        self,
        images: ImageSeq,
        /,
        masks: MaskSeqOrNone = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        **kwargs: typing.Any,
    ) -> None:
        return None

    def process_directories(
        self,
        image_dir: intnormt.PathLike,
        /,
        mask_dir: intnormt.PathLike | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        ext: str = "nii*",
        return_normalized_and_masks: bool = False,
        **kwargs: typing.Any,
    ) -> tuple[ImageSeq, MaskSeqOrNone] | None:
        logger.debug("Grabbing images")
        images, masks = intnormio.gather_images_and_masks(image_dir, mask_dir, ext=ext)
        self.fit(images, masks, modality=modality, **kwargs)
        if return_normalized_and_masks:
            normalized: list[intnormt.ImageLike] = []
            n_images = len(images)
            zipped = intnormio.zip_with_nones(images, masks)
            for i, (image, mask) in enumerate(zipped, 1):
                logger.info(f"Normalizing image {i}/{n_images}")
                normalized.append(self(image, mask, modality=modality))
            return normalized, masks
        return None

    def plot_histogram_from_args(
        self,
        args: argparse.Namespace,
        /,
        normalized: ImageSeq,
        masks: MaskSeqOrNone = None,
    ) -> None:
        import matplotlib.pyplot as plt

        import intensity_normalization.plot.histogram as intnormhist

        if args.output_dir is None:
            output = pathlib.Path(args.image_dir) / "hist.pdf"
        else:
            output = pathlib.Path(args.output_dir) / "hist.pdf"
        hp = intnormhist.HistogramPlotter(title=self.fullname())
        _ = hp(normalized, masks)
        plt.savefig(output)

    def call_from_argparse_args(
        self,
        args: argparse.Namespace,
        /,
        *,
        use_masks_in_plot: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        out = self.process_directories(
            args.image_dir,
            args.mask_dir,
            modality=intnormt.Modality.from_string(args.modality),
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
            norm_image.view(mioi.Image).to_filename(fn)
        self.save_additional_info(
            args,
            normalized=normalized,
            masks=masks,
            image_filenames=image_filenames,
        )
        if args.plot_histogram:
            _masks = masks if use_masks_in_plot else None
            self.plot_histogram_from_args(args, normalized, _masks)


class DirectoryNormalizeCLI(
    SampleNormalizeCLIMixin, intnormcli.DirectoryCLI, metaclass=abc.ABCMeta
):
    def fit(
        self,
        images: ImageSeq,
        /,
        masks: MaskSeqOrNone = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        **kwargs: typing.Any,
    ) -> None:
        images, masks = self.before_fit(images, masks, modality=modality, **kwargs)
        logger.info("Fitting")
        self._fit(images, masks, modality=modality, **kwargs)
        logger.debug("Done fitting")

    def _fit(
        self,
        images: ImageSeq,
        /,
        masks: MaskSeqOrNone = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        **kwargs: typing.Any,
    ) -> None:
        raise NotImplementedError

    def before_fit(
        self,
        images: collections.abc.Sequence[intnormt.ImageLike],
        /,
        masks: collections.abc.Sequence[intnormt.ImageLike] | None = None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        **kwargs: typing.Any,
    ) -> tuple[ImageSeq, MaskSeqOrNone]:
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
        modality: intnormt.Modality = intnormt.Modality.T1,
        ext: str = "nii*",
        return_normalized_and_masks: bool = False,
        **kwargs: typing.Any,
    ) -> tuple[ImageSeq, MaskSeqOrNone] | None:
        return self.process_directories(
            image_dir,
            mask_dir,
            modality=modality,
            ext=ext,
            return_normalized_and_masks=return_normalized_and_masks,
            **kwargs,
        )
