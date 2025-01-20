"""CLI for Nyul & Udupa normalization."""

__all__ = ["nyul_main", "nyul_parser"]

from intensity_normalization.normalize.nyul import NyulNormalize

# main functions and parsers for CLI
nyul_parser = NyulNormalize.parser()
nyul_main = NyulNormalize.main(nyul_parser)
