"""CLI for RAVEL normalization
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 13 Oct 2021
"""

__all__ = ["ravel_main", "ravel_parser"]

from intensity_normalization.normalize.ravel import RavelNormalize

# main functions and parsers for CLI
ravel_parser = RavelNormalize.parser()
ravel_main = RavelNormalize.main(ravel_parser)
