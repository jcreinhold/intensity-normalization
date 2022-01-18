# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.preprocess

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["preprocess_main", "preprocess_parser"]

from intensity_normalization.util.preprocess import Preprocessor

# main functions and parsers for CLI
preprocess_parser = Preprocessor.parser()
preprocess_main = Preprocessor.main(preprocess_parser)
