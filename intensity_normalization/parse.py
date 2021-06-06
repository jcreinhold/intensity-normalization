#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity-normalization.cli.parse

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 06, 2021
"""

__all__ = [
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

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, List, Union


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
            raise argparse.ArgumentTypeError(msg)
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
            raise argparse.ArgumentTypeError(msg)
        return str(path.resolve())


class positive_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.0:
            msg = f"{string} needs to be a positive float."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            msg = f"{string} needs to be a positive integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_odd_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[int, None]:
        num = int(string)
        if num <= 0 or not (num % 2):
            msg = f"{string} needs to be a positive odd integer."
            raise argparse.ArgumentTypeError(msg)
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
            raise argparse.ArgumentTypeError(msg)
        return num


class nonnegative_float(_ParseType):
    def __call__(self, string: str) -> int:
        num = float(string)
        if num < 0.0:
            msg = f"{string} needs to be a nonnegative float."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0 or num > 1.0:
            msg = f"{string} needs to be between 0 and 1."
            raise argparse.ArgumentTypeError(msg)
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


def remove_args(parser: argparse.ArgumentParser, args: List[str]):
    """ remove a list of arguments from a parser """
    # https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse
    for arg in args:
        for action in parser._actions:
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
