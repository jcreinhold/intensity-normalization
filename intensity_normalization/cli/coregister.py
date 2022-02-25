"""Co-registration CLI
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: 13 Oct 2021
"""

__all__ = ["coregister_main", "coregister_parser"]

from intensity_normalization.util.coregister import Registrator

# main functions and parsers for CLI
coregister_parser = Registrator.parser()
coregister_main = Registrator.main(coregister_parser)
