#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity-normalization.cli.parse

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 06, 2021
"""

__all__ = [
    "CLI",
    "dir_path",
    "file_path",
    "new_parse_type",
    "nonnegative_float",
    "nonnegative_int",
    "positive_float",
    "positive_int",
    "positive_int_or_none",
    "positive_odd_int_or_none",
    "probability_float",
    "probability_float_or_none",
    "save_file_path",
    "save_nifti_path",
    "setup_log",
]

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar, Union

import nibabel as nib

from intensity_normalization.type import ArgType, NiftiImage, PathLike
from intensity_normalization.util.io import split_filename


def return_none(func: Callable) -> Callable:
    def new_func(self, string: Any) -> Any:  # type: ignore[no-untyped-def]
        if string is None:
            return None
        elif isinstance(string, str):
            if string.lower() in ("none", "null"):
                return None
        return func(self, string)

    return new_func


class _ParseType:
    @property
    def __name__(self) -> str:
        name = self.__class__.__name__
        assert isinstance(name, str)
        return name

    def __str__(self) -> str:
        return self.__name__


class save_file_path(_ParseType):
    def __call__(self, string: str) -> Path:
        if not string.isprintable():
            msg = "String must only contain printable characters."
            raise ArgumentTypeError(msg)
        path = Path(string)
        return path


class save_nifti_path(_ParseType):
    def __call__(self, string: str) -> Path:
        not_nifti = not string.endswith(".nii.gz") and not string.endswith(".nii")
        if not_nifti or not string.isprintable():
            msg = (
                f"{string} is not a valid path to a NIfTI file. "
                "Needs to end with .nii or .nii.gz and can "
                "only contain printable characters."
            )
            raise ArgumentTypeError(msg)
        path = Path(string)
        return path


class dir_path(_ParseType):
    def __call__(self, string: str) -> str:
        path = Path(string)
        if not path.is_dir():
            msg = f"{string} is not a valid directory path."
            raise ArgumentTypeError(msg)
        return str(path.resolve())


class file_path(_ParseType):
    def __call__(self, string: str) -> str:
        path = Path(string)
        if not path.is_file():
            msg = f"{string} is not a valid file path."
            raise ArgumentTypeError(msg)
        return str(path.resolve())


class positive_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.0:
            msg = f"{string} needs to be a positive float."
            raise ArgumentTypeError(msg)
        return num


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            msg = f"{string} needs to be a positive integer."
            raise ArgumentTypeError(msg)
        return num


class positive_odd_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[int, None]:
        num = int(string)
        if num <= 0 or not (num % 2):
            msg = f"{string} needs to be a positive odd integer."
            raise ArgumentTypeError(msg)
        return num


class positive_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[int, None]:
        return positive_int()(string)


class nonnegative_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num < 0:
            msg = f"{string} needs to be a nonnegative integer."
            raise ArgumentTypeError(msg)
        return num


class nonnegative_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0:
            msg = f"{string} needs to be a nonnegative float."
            raise ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0 or num > 1.0:
            msg = f"{string} needs to be between 0 and 1."
            raise ArgumentTypeError(msg)
        return num


class probability_float_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[float, None]:
        return probability_float()(string)


class NewParseType:
    def __init__(self, func: Callable, name: str):
        self.name = name
        self.func = func

    def __str__(self) -> str:
        return self.name

    def __call__(self, val: Any) -> Any:
        return self.func(val)


def new_parse_type(func: Callable, name: str) -> NewParseType:
    return NewParseType(func, name)


def setup_log(verbosity: int) -> None:
    """ set logger with verbosity logging level and message """
    if verbosity == 1:
        level = logging.getLevelName("INFO")
    elif verbosity >= 2:
        level = logging.getLevelName("DEBUG")
    else:
        level = logging.getLevelName("WARNING")
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=level)
    logging.captureWarnings(True)


T = TypeVar("T", bound="CLI")


class CLI:
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

    def append_name_to_file(
        self, filepath: PathLike, alternate_path: Optional[PathLike] = None,
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
    def parser(cls: Type[T]) -> ArgumentParser:
        parser = cls.get_parent_parser(cls.description())
        parser = cls.add_method_specific_arguments(parser)
        return parser

    @classmethod
    def main(cls: Type[T], parser: ArgumentParser) -> Callable:
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
    def from_argparse_args(cls: Type[T], args: Namespace) -> T:
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
        out.to_filename(args.output)

    @staticmethod
    def load_image(image_path: PathLike) -> NiftiImage:
        return nib.load(image_path)
