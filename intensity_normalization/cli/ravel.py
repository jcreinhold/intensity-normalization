# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.ravel

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["ravel_main", "ravel_parser"]

import sys

try:
    import ants
except (ModuleNotFoundError, ImportError):
    print("ANTsPy not installed. Install antspyx to use ravel-normalize.")
    sys.exit(1)

from intensity_normalization.normalize.ravel import RavelNormalize

# main functions and parsers for CLI
ravel_parser = RavelNormalize.parser()
ravel_main = RavelNormalize.main(ravel_parser)
