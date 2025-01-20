"""Least-squares fit tissue mean normalization."""

__all__ = ["lsq_main", "lsq_parser"]

from intensity_normalization.normalize.lsq import LeastSquaresNormalize

# main functions and parsers for CLI
lsq_parser = LeastSquaresNormalize.parser()
lsq_main = LeastSquaresNormalize.main(lsq_parser)
