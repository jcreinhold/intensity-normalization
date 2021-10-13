# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.kde

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["kde_main", "kde_parser"]

from intensity_normalization.normalize.kde import KDENormalize

# main functions and parsers for CLI
kde_parser = KDENormalize.parser()
kde_main = KDENormalize.main(kde_parser)
