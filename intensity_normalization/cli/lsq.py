# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.lsq

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["lsq_main", "lsq_parser"]

from intensity_normalization.normalize.lsq import LeastSquaresNormalize

# main functions and parsers for CLI
lsq_parser = LeastSquaresNormalize.parser()
lsq_main = LeastSquaresNormalize.main(lsq_parser)
