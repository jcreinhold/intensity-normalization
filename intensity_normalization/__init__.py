"""Top-level package for intensity-normalization."""

import logging

__title__ = "intensity-normalization"
__description__ = "normalize the intensities of various MR image modalities"
__url__ = "https://github.com/jcreinhold/intensity-normalization"
__author__ = """Jacob Reinhold"""
__email__ = "jcreinhold@gmail.com"
__version__ = "2.1.2"
__license__ = "Apache-2.0"
__copyright__ = "Copyright 2021 Jacob Reinhold"

PEAK = {
    "last": ["t1", "other", "last"],
    "largest": ["t2", "flair", "largest"],
    "first": ["pd", "md", "first"],
}
VALID_PEAKS = {m for modalities in PEAK.values() for m in modalities}
VALID_MODALITIES = VALID_PEAKS - {"last", "largest", "first"}

logging.getLogger(__name__).addHandler(logging.NullHandler())
