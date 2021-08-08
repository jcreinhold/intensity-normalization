"""Top-level package for intensity-normalization."""

__author__ = """Jacob Reinhold"""
__email__ = "jcreinhold@gmail.com"
__version__ = "2.0.0"

PEAK = {
    "last": ["t1", "last"],
    "largest": ["t2", "flair", "largest"],
    "first": ["pd", "md", "first"],
}
VALID_PEAKS = {m for modalities in PEAK.values() for m in modalities}
VALID_MODALITIES = VALID_PEAKS - {"last", "largest", "first"}
