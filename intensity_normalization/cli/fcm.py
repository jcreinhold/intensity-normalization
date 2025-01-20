"""CLI for fuzzy c-means tissue mean normalization."""

__all__ = ["fcm_main", "fcm_parser"]

from intensity_normalization.normalize.fcm import FCMNormalize

# main functions and parsers for CLI
fcm_parser = FCMNormalize.parser()
fcm_main = FCMNormalize.main(fcm_parser)
