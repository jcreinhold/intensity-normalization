"""CLI base class for normalization/preprocessing methods
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 06 Jun 2021
"""

from __future__ import annotations

__all__ = ["CLIMixin", "DirectoryCLI", "setup_log", "SingleImageCLI"]

import abc
import argparse
import logging
import pathlib
import sys
import typing

import pymedio.image as mioi

import intensity_normalization as intnorm
import intensity_normalization.typing as intnormt
import intensity_normalization.util.io as intnormio
from intensity_normalization import __version__ as int_norm_version

logger = logging.getLogger(__name__)

T = typing.TypeVar("T")


def setup_log(verbosity: int) -> None:
    """set logger with verbosity logging level and message"""
    if verbosity == 1:
        level = logging.getLevelName("INFO")
    elif verbosity >= 2:
        level = logging.getLevelName("DEBUG")
    else:
        level = logging.getLevelName("WARNING")
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=level)
    logging.captureWarnings(True)


class CLIMixin(metaclass=abc.ABCMeta):
    def __str__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def fullname() -> str:
        raise NotImplementedError

    def append_name_to_file(
        self,
        filepath: intnormt.PathLike,
        alternate_path: intnormt.PathLike | None = None,
    ) -> pathlib.Path:
        path, base, ext = intnormio.split_filename(filepath)
        if alternate_path is not None:
            path = pathlib.Path(alternate_path).resolve()
            assert path.is_dir()
        new_path: pathlib.Path = path / (base + f"_{self.name()}" + ext)
        return new_path

    @classmethod
    @abc.abstractmethod
    def get_parent_parser(
        cls,
        desc: str,
        valid_modalities: frozenset[str] = intnorm.VALID_MODALITIES,
        **kwargs: typing.Any,
    ) -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def add_method_specific_arguments(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        return parent_parser

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        parser = cls.get_parent_parser(cls.description())
        parser = cls.add_method_specific_arguments(parser)
        return parser

    @classmethod
    def main(
        cls, parser: argparse.ArgumentParser
    ) -> typing.Callable[[intnormt.ArgType], int]:
        def _main(args: intnormt.ArgType = None) -> int:
            if args is None:
                if len(sys.argv) == 2 and sys.argv[1] == "--version":
                    print(f"intensity-normalization version {int_norm_version}")
                    return 0
                args = parser.parse_args()
            elif isinstance(args, list):
                args = parser.parse_args(args)
            else:
                raise ValueError("args must be None or a list of strings to parse")
            if args.version:
                print(f"intensity-normalization version {int_norm_version}")
            setup_log(args.verbosity)
            cls_instance = cls.from_argparse_args(args)
            cls_instance.call_from_argparse_args(args)
            return 0

        return _main

    @classmethod
    @abc.abstractmethod
    def from_argparse_args(cls: typing.Type[T], args: argparse.Namespace) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def call_from_argparse_args(
        self, args: argparse.Namespace, /, **kwargs: typing.Any
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def load_image(image_path: intnormt.PathLike) -> mioi.Image:
        return mioi.Image.from_path(image_path)


class SingleImageCLI(CLIMixin, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        image: intnormt.ImageLike,
        /,
        mask: intnormt.ImageLike | None,
        *,
        modality: intnormt.Modality = intnormt.Modality.T1,
        **kwargs: typing.Any,
    ) -> typing.Any:
        raise NotImplementedError

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
            help="Path of image to process.",
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
            help="Path to save the processed image.",
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

    def call_from_argparse_args(
        self, args: argparse.Namespace, /, **kwargs: typing.Any
    ) -> None:
        image = self.load_image(args.image)
        mask: intnormt.ImageLike | None
        if hasattr(args, "mask") and args.mask is not None:
            mask = self.load_image(args.mask)
        else:
            mask = None
        out = self(image, mask)
        if args.output is None:
            args.output = self.append_name_to_file(args.image)
        logger.debug(f"Saving output: {args.output}")
        if hasattr(out, "save"):
            out.save(args.output)
        elif hasattr(out, "to_filename"):
            out.to_filename(args.output)
        else:
            raise ValueError("Unexpected image type")


class DirectoryCLI(CLIMixin, metaclass=abc.ABCMeta):
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
            "image_dir",
            type=intnormt.dir_path(),
            help="Path of directory containing images to normalize.",
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
            default="t1",
            choices=intnorm.VALID_MODALITIES,
            help="Modality of the images.",
        )
        parser.add_argument(
            "-e",
            "--extension",
            type=str,
            default="nii*",
            help="Extension of images.",
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

    @abc.abstractmethod
    def call_from_argparse_args(
        self, args: argparse.Namespace, /, **kwargs: typing.Any
    ) -> None:
        raise NotImplementedError
