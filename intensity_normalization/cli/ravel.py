"""CLI for RAVEL normalization."""

__all__ = ["ravel_main", "ravel_parser"]

from intensity_normalization.normalize.ravel import RavelNormalize

# main functions and parsers for CLI
ravel_parser = RavelNormalize.parser()
ravel_main = RavelNormalize.main(ravel_parser)
