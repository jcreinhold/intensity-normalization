"""Exceptions for intensity-normalization.

Runtime failures raise :class:`IntensityNormalizationError` with a message that
says what to do about it. Bad arguments (wrong types, out-of-range values)
surface as ordinary ``TypeError``/``ValueError``.
"""

from __future__ import annotations

__all__ = ["IntensityNormalizationError"]


class IntensityNormalizationError(Exception):
    """A normalization operation could not be completed."""
