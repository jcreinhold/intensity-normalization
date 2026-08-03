"""Shared math for the method cores (private).

Every normalization ends in one affine map of the intensities: estimate
reference statistics from the foreground, then standardize the array by them.
That application step lives here once, dtype discipline included (float64 in
-> float64 out; float32 otherwise).
"""

from __future__ import annotations

from intensity_normalization._image import IntensityArray

__all__ = ["standardize"]


def standardize(data: IntensityArray, center: float, spread: float, norm_value: float) -> IntensityArray:
    """``(data - center) / spread * norm_value``, stored in ``data``'s dtype.

    Spread validity is the *caller's* check: the zero-spread error must name
    the statistic that failed ("white stripe has zero standard deviation"),
    so each core validates before calling and keeps its actionable message.

    Memory: one volume-sized allocation. ``data - center`` allocates the
    output (never mutates ``data`` — ``unwrap`` may hand back the caller's
    own array), the remaining steps run in place, and Python-scalar
    arithmetic keeps ``data``'s dtype (NEP 50 weak promotion), so no
    ``astype`` copy is needed.
    """
    out = data - center
    out /= spread
    out *= norm_value
    return out
