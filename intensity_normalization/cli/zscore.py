# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.zscore

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["zscore_main", "zscore_parser"]

from intensity_normalization.normalize.zscore import ZScoreNormalize

# main functions and parsers for CLI
zscore_parser = ZScoreNormalize.parser()
zscore_main = ZScoreNormalize.main(zscore_parser)
