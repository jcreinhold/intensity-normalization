"""Co-registration CLI."""

__all__ = ["coregister_main", "coregister_parser"]

from intensity_normalization.util.coregister import Registrator

# main functions and parsers for CLI
coregister_parser = Registrator.parser()
coregister_main = Registrator.main(coregister_parser)
