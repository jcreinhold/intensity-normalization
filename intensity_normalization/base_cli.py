"""CLI base class for normalization/preprocessing methods

process command-line arguments for
normalization or other utility scripts

Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 06 Jun 2021
"""

from __future__ import annotations

__all__ = ["CLI", "setup_log"]

import abc
import argparse
import builtins
import logging
import pathlib
import sys
import typing

import pymedio.image as mioi

import intensity_normalization.typing as intnormt
import intensity_normalization.util.io as intnormio
from intensity_normalization import __version__ as int_norm_version

logger = logging.getLogger(__name__)


def setup_log(verbosity: builtins.int) -> None:
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


class CLI(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        image: intnormt.Image,
        /,
        mask: intnormt.Image | None,
        *,
        modality: intnormt.Modalities = intnormt.Modalities.T1,
        **kwargs,
    ):
        raise NotImplementedError

    def __str__(self) -> builtins.str:
        return self.__class__.__name__

    @staticmethod
    @abc.abstractmethod
    def description() -> builtins.str:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def name() -> builtins.str:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def fullname() -> builtins.str:
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
        return path / (base + f"_{self.name()}" + ext)

    @staticmethod
    @abc.abstractmethod
    def get_parent_parser(desc: builtins.str, **kwargs) -> argparse.ArgumentParser:
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
    ) -> typing.Callable[[intnormt.ArgType], builtins.int]:
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
    def from_argparse_args(cls, args: argparse.Namespace) -> CLI:
        raise NotImplementedError

    def call_from_argparse_args(self, args: argparse.Namespace) -> None:
        image = self.load_image(args.image)
        if hasattr(args, "mask"):
            mask = args.mask and self.load_image(args.mask)
        else:
            mask = None
        out = self(image, mask)
        if args.output is None:
            args.output = self.append_name_to_file(args.image)
        logger.debug(f"Saving output: {args.output}")
        out.to_filename(args.output)

    @staticmethod
    def load_image(image_path: intnormt.PathLike) -> mioi.Image:
        return mioi.Image.from_path(image_path)
