"""CLI for WhiteStripe normalization method."""

__all__ = ["whitestripe_main", "whitestripe_parser"]

from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize

# main functions and parsers for CLI
whitestripe_parser = WhiteStripeNormalize.parser()
whitestripe_main = WhiteStripeNormalize.main(whitestripe_parser)
