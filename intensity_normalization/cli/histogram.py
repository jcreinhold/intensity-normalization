"""CLI for plotting histograms of a set of images."""

__all__ = ["histogram_main", "histogram_parser"]

from intensity_normalization.plot.histogram import HistogramPlotter

# main functions and parsers for CLI
histogram_parser = HistogramPlotter.parser()
histogram_main = HistogramPlotter.main(histogram_parser)
