#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity-normalization.cli.fcm_normalize

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 06, 2021
"""

__all__ = []

import argparse
import logging

from intensity_normalization.parse import setup_log
from intensity_normalization.type import ArgType
from intensity_normalization.normalize.fcm import FCMNormalize


def single_image_parser() -> argparse.ArgumentParser:
    desc = (
        "Use fuzzy c-means to find memberships of CSF/GM/WM in the brain. "
        "Use the found and specified tissue mean to normalize a NIfTI MRI."
    )
    parser = FCMNormalize.get_parent_parser(desc)
    parser = FCMNormalize.add_method_specific_arguments(parser)
    return parser


def single_image_main(args: ArgType = None) -> int:
    parser = single_image_parser()
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, list):
        args = parser.parse_args(args)
    else:
        raise ValueError("args must be None or a list of strings to parse")
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    fcm_normalizer = FCMNormalize.from_argparse_args(args)
    fcm_normalizer.normalize_from_argparse_args(args)
    return 0


def directory_parser() -> argparse.ArgumentParser:
    raise NotImplementedError


def fcm_directory(args: ArgType = None) -> int:
    raise NotImplementedError
