"""KDE-based normalization: scale the tissue mode of the smoothed histogram to a fixed value."""

from __future__ import annotations

from intensity_normalization import _image, histogram
from intensity_normalization._image import Image, IntensityArray, Mask
from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["kde", "kde_array"]


def kde_array(
    data: IntensityArray,
    mask: IntensityArray | None = None,
    *,
    modality: str = "t1",
    peak: histogram.Peak | None = None,
    norm_value: float = 1.0,
    seed: int | None = 0,
) -> IntensityArray:
    """Normalize an intensity array by the tissue mode of its smoothed histogram.

    Fits a kernel density estimate to the foreground intensities, finds the
    mode of the tissue of interest (white matter by default for T1-w), and
    scales the array so that mode equals ``norm_value``.

    Args:
        data: intensity array.
        mask: foreground (brain) mask array. If None, estimated as positive voxels.
        modality: one of "t1", "t2", "flair", "pd", "md", "other"; selects
            which histogram peak is the tissue of interest.
        peak: explicit peak override ("last", "largest", "first") for
            non-standard data.
        norm_value: intensity the tissue mode is mapped to.
        seed: RNG seed for the KDE subsample; ``None`` is nondeterministic.

    Returns:
        The normalized intensity array.
    """
    foreground = _image.foreground_values(data, mask)
    mode = histogram.tissue_mode(foreground, modality=modality, peak=peak, seed=seed)
    if mode == 0.0:
        msg = "The tissue mode is at zero intensity; cannot scale by it. Check the image and mask."
        raise IntensityNormalizationError(msg)
    return data / mode * norm_value


def kde(
    image: Image,
    mask: Mask | None = None,
    *,
    modality: str = "t1",
    peak: histogram.Peak | None = None,
    norm_value: float = 1.0,
    seed: int | None = 0,
) -> Image:
    """Normalize an MR image (numpy or nibabel); see :func:`kde_array`.

    Args:
        image: numpy array or nibabel image; the same type is returned.
        mask: foreground (brain) mask. If None, estimated as positive voxels.
        modality: one of "t1", "t2", "flair", "pd", "md", "other"; selects
            which histogram peak is the tissue of interest.
        peak: explicit peak override ("last", "largest", "first") for
            non-standard data.
        norm_value: intensity the tissue mode is mapped to.
        seed: RNG seed for the KDE subsample; ``None`` is nondeterministic.

    Returns:
        The normalized image, same type as ``image``.
    """
    data, restore = _image.unwrap(image)
    normalized = kde_array(
        data, _image.unwrap_mask(image, mask), modality=modality, peak=peak, norm_value=norm_value, seed=seed
    )
    return restore(normalized)
