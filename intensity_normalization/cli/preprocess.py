# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.preprocess

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["preprocess_main", "preprocess_parser"]

import sys

try:
    import ants
except (ModuleNotFoundError, ImportError):
    print("ANTsPy not installed. Install antspyx to use preprocess.")
    sys.exit(1)

from intensity_normalization.util.preprocess import Preprocessor

# main functions and parsers for CLI
preprocess_parser = Preprocessor.parser()
preprocess_main = Preprocessor.main(preprocess_parser)
