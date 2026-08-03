"""Histogram estimation and tissue modes of MR image intensities.

Shared by the KDE, WhiteStripe, and LSQ methods, and public so users can
inspect histograms when validating normalization results.

The modality -> tissue-mode policy lives here: for T1-w images white matter is
the highest-intensity tissue peak ("last"); for T2-w/FLAIR the interesting
tissue is the global maximum ("largest"); for PD/MD it is the lowest peak
("first"). Non-standard data can override the policy with an explicit ``peak``.
"""

from __future__ import annotations

import typing

import numpy as np
import scipy.signal
import scipy.stats

from intensity_normalization._image import ForegroundIntensities, IntensityArray
from intensity_normalization.errors import IntensityNormalizationError

__all__ = [
    "MODALITY_PEAKS",
    "VALID_MODALITIES",
    "VALID_PEAKS",
    "first_mode",
    "largest_mode",
    "last_mode",
    "smooth_histogram",
    "tissue_mode",
]

Peak = typing.Literal["last", "largest", "first"]  # plain assignment: typing.get_args(Peak) is used at runtime

VALID_PEAKS: tuple[Peak, ...] = typing.get_args(Peak)

#: Which histogram peak carries the tissue of interest, per modality.
MODALITY_PEAKS: dict[str, Peak] = {
    "t1": "last",
    "t2": "largest",
    "flair": "largest",
    "pd": "first",
    "md": "first",
    "other": "last",
}

VALID_MODALITIES: tuple[str, ...] = tuple(MODALITY_PEAKS)

_GRID_SIZE = 80
_MAX_KDE_SAMPLES = 50_000


def smooth_histogram(
    intensities: ForegroundIntensities, *, max_samples: int = _MAX_KDE_SAMPLES, seed: int | None = 0
) -> tuple[IntensityArray, IntensityArray]:
    """Kernel density estimate of the intensity distribution.

    Uses a seeded subsample of at most ``max_samples`` intensities: the KDE is
    O(n) in the sample count per grid point and the mode is statistically
    unchanged, so this keeps the estimate fast on multi-million-voxel images.

    Args:
        intensities: 1D array of (foreground) intensities.
        max_samples: cap on the number of samples fed to the KDE.
        seed: subsample RNG seed; ``None`` is nondeterministic.

    Returns:
        ``(grid, pdf)``: the intensity grid and the estimated density on it.
    """
    x = np.asarray(intensities, dtype=np.float64).ravel()
    if x.size == 0:
        msg = "Cannot estimate a histogram of zero intensities (empty foreground?)."
        raise IntensityNormalizationError(msg)
    if x.min() == x.max():
        msg = (
            f"Foreground intensities are constant (all equal to {x.min():.4g}); "
            "cannot estimate a histogram. Check the image and mask."
        )
        raise IntensityNormalizationError(msg)
    if x.size > max_samples:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=max_samples, replace=False)
    try:
        kde = scipy.stats.gaussian_kde(x)
    except scipy.linalg.LinAlgError as exn:
        msg = (
            "Could not estimate a smooth histogram of the foreground "
            "(near-constant intensities?). Check the image and mask."
        )
        raise IntensityNormalizationError(msg) from exn
    grid = np.linspace(x.min(), x.max(), _GRID_SIZE)
    return grid, kde(grid)


def largest_mode(intensities: ForegroundIntensities, **kwargs: typing.Any) -> float:
    """Mode of the largest tissue class (global maximum of the smoothed histogram)."""
    grid, pdf = smooth_histogram(intensities, **kwargs)
    return float(grid[np.argmax(pdf)])


def last_mode(intensities: ForegroundIntensities, *, tail_percentage: float = 96.0, **kwargs: typing.Any) -> float:
    """Mode of the highest-intensity tissue class (last local maximum of the histogram).

    The histogram above ``tail_percentage`` is removed first, so bright tails
    (e.g., vessels, lesions, fat) do not count as the tissue mode.
    """
    if not 0.0 < tail_percentage < 100.0:
        raise ValueError(f"tail_percentage must be in (0, 100). Got {tail_percentage}.")
    x = np.asarray(intensities, dtype=np.float64).ravel()
    x = x[x <= np.percentile(x, tail_percentage)]
    grid, pdf = smooth_histogram(x, **kwargs)
    maxima = scipy.signal.argrelmax(pdf)[0]
    if maxima.size == 0:
        return float(grid[np.argmax(pdf)])
    return float(grid[maxima[-1]])


def first_mode(intensities: ForegroundIntensities, *, tail_percentage: float = 99.0, **kwargs: typing.Any) -> float:
    """Mode of the lowest-intensity tissue class (first local maximum of the histogram)."""
    if not 0.0 < tail_percentage < 100.0:
        raise ValueError(f"tail_percentage must be in (0, 100). Got {tail_percentage}.")
    x = np.asarray(intensities, dtype=np.float64).ravel()
    x = x[x <= np.percentile(x, tail_percentage)]
    grid, pdf = smooth_histogram(x, **kwargs)
    maxima = scipy.signal.argrelmax(pdf)[0]
    if maxima.size == 0:
        return float(grid[np.argmax(pdf)])
    return float(grid[maxima[0]])


def tissue_mode(
    intensities: ForegroundIntensities, *, modality: str = "t1", peak: Peak | None = None, **kwargs: typing.Any
) -> float:
    """Mode of the tissue of interest for ``modality`` (or an explicit ``peak``).

    Args:
        intensities: 1D array of foreground intensities.
        modality: one of "t1", "t2", "flair", "pd", "md", "other".
        peak: explicit peak override ("last", "largest", "first") for
            non-standard data; derived from ``modality`` when None.
    """
    if peak is None:
        modality = modality.lower()
        if modality not in MODALITY_PEAKS:
            choices = ", ".join(VALID_MODALITIES)
            raise ValueError(f"Unknown modality {modality!r}. Choose one of: {choices}.")
        peak = MODALITY_PEAKS[modality]
    elif peak not in VALID_PEAKS:
        raise ValueError(f"Unknown peak {peak!r}. Choose one of: {', '.join(VALID_PEAKS)}.")
    if peak == "last":
        return last_mode(intensities, **kwargs)
    if peak == "largest":
        return largest_mode(intensities, **kwargs)
    return first_mode(intensities, **kwargs)
