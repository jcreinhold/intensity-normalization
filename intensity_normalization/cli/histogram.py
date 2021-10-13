# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.histogram

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["histogram_main", "histogram_parser"]

from intensity_normalization.plot.histogram import HistogramPlotter

# main functions and parsers for CLI
histogram_parser = HistogramPlotter.parser()
histogram_main = HistogramPlotter.main(histogram_parser)
