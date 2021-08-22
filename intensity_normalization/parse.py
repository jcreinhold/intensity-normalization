#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity-normalization.parse

process command-line arguments for
normalization or other utility scripts

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 06, 2021
"""

__all__ = [
    "CLIParser",
    "setup_log",
]

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable, Optional, Type, TypeVar

import nibabel as nib

from intensity_normalization.type import ArgType, NiftiImage, PathLike
from intensity_normalization.util.io import split_filename

logger = logging.getLogger(__name__)


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


CP = TypeVar("CP", bound="CLIParser")


class CLIParser:
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def description() -> str:
        raise NotImplementedError

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    def fullname() -> str:
        raise NotImplementedError

    def append_name_to_file(
        self,
        filepath: PathLike,
        alternate_path: Optional[PathLike] = None,
    ) -> Path:
        path, base, ext = split_filename(filepath)
        if alternate_path is not None:
            path = Path(alternate_path).resolve()
            assert path.is_dir()
        return path / (base + f"_{self.name()}" + ext)

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def add_method_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        return parent_parser

    @classmethod
    def parser(cls: Type[CP]) -> ArgumentParser:
        parser = cls.get_parent_parser(cls.description())
        parser = cls.add_method_specific_arguments(parser)
        return parser

    @classmethod
    def main(cls: Type[CP], parser: ArgumentParser) -> Callable:
        def _main(args: ArgType = None) -> int:
            if args is None:
                args = parser.parse_args()
            elif isinstance(args, list):
                args = parser.parse_args(args)
            else:
                raise ValueError("args must be None or a list of strings to parse")
            setup_log(args.verbosity)
            cls_instance = cls.from_argparse_args(args)
            cls_instance.call_from_argparse_args(args)
            return 0

        return _main

    @classmethod
    def from_argparse_args(cls: Type[CP], args: Namespace) -> CP:
        raise NotImplementedError

    def call_from_argparse_args(self, args: Namespace) -> None:
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
    def load_image(image_path: PathLike) -> NiftiImage:
        return nib.load(image_path)
