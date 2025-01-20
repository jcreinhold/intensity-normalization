"""Kernel density estimate tissue mode normalization CLI."""

__all__ = ["kde_main", "kde_parser"]

from intensity_normalization.normalize.kde import KDENormalize

# main functions and parsers for CLI
kde_parser = KDENormalize.parser()
kde_main = KDENormalize.main(kde_parser)
