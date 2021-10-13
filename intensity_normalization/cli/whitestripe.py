# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.whitestripe

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["whitestripe_main", "whitestripe_parser"]

from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize

# main functions and parsers for CLI
whitestripe_parser = WhiteStripeNormalize.parser()
whitestripe_main = WhiteStripeNormalize.main(whitestripe_parser)
