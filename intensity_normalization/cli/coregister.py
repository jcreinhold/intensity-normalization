# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.coregister

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["coregister_main", "coregister_parser"]

import sys

try:
    import ants
except (ModuleNotFoundError, ImportError):
    print("ANTsPy not installed. Install antspyx to use coregister.")
    sys.exit(1)

from intensity_normalization.util.coregister import Registrator

# main functions and parsers for CLI
coregister_parser = Registrator.parser()
coregister_main = Registrator.main(coregister_parser)
