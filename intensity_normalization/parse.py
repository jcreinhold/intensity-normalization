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
    "remove_args",
    "save_file_path",
    "setup_log",
]

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from pathlib import Path
from typing import Any, Callable, List, Union

import nibabel as nib

from intensity_normalization.type import ArgType, NiftiImage, PathLike
from intensity_normalization.util.io import split_filename


def return_none(func: Callable) -> Callable:
    def new_func(self, string) -> Any:
        if string is None:
            return None
        elif isinstance(string, str):
            if string.lower() in ("none", "null"):
                return None
            else:
                return func(self, string)
        else:
            return func(self, string)

    return new_func


class _ParseType:
    @property
    def __name__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__name__


class save_file_path(_ParseType):
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
            raise argparse.ArgumentTypeError(msg)
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
    def __call__(self, string: str) -> int:
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


def new_parse_type(func: Callable, name: str):
    class NewParseType:
        def __str__(self):
            return name

        def __call__(self, val: Any):
            return func(val)

    return NewParseType()


def remove_args(parser: ArgumentParser, args: List[str]):
    """ remove a list of arguments from a parser """
    # https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse
    for arg in args:
        for action in parser._actions:
            if len(action.option_strings) == 0:
                continue
            opt_str = action.option_strings[-1]
            dest = action.dest
            if opt_str[0] == arg or dest == arg:
                parser._remove_action(action)
                break

        for action in parser._action_groups:
            group_actions = action._group_actions
            for group_action in group_actions:
                if group_action.dest == arg:
                    group_actions.remove(group_action)
                    break


def setup_log(verbosity: int):
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


class CLI:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def description() -> str:
        raise NotImplementedError

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    def append_name_to_file(self, filepath: PathLike) -> Path:
        path, base, ext = split_filename(filepath)
        return path / (base + f"_{self.name()}" + ext)

    @staticmethod
    def get_parent_parser(desc: str) -> ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def add_method_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        return parent_parser

    @classmethod
    def parser(cls) -> ArgumentParser:
        parser = cls.get_parent_parser(cls.description())
        parser = cls.add_method_specific_arguments(parser)
        return parser

    @classmethod
    def main(cls, parser: ArgumentParser) -> Callable:
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
    def from_argparse_args(cls, args: Namespace):
        raise NotImplementedError

    def call_from_argparse_args(self, args: Namespace):
        image = self.load_image(args.image)
        mask = self.load_image(args.mask) if hasattr(args, "mask") else None
        out = self(image, mask)
        if args.output is None:
            args.output = self.append_name_to_file(args.image)
        out.to_filename(args.output)

    @staticmethod
    def load_image(image_path: PathLike) -> NiftiImage:
        return nib.load(image_path)
