# -*- coding: utf-8 -*-
"""
intensity_normalization.cli.fcm

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 13 Oct 2021
"""

__all__ = ["fcm_main", "fcm_parser"]

from intensity_normalization.normalize.fcm import FCMNormalize

# main functions and parsers for CLI
fcm_parser = FCMNormalize.parser()
fcm_main = FCMNormalize.main(fcm_parser)
