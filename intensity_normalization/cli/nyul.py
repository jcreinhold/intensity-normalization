# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.nyul

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["nyul_main", "nyul_parser"]

from intensity_normalization.normalize.nyul import NyulNormalize

# main functions and parsers for CLI
nyul_parser = NyulNormalize.parser()
nyul_main = NyulNormalize.main(nyul_parser)
