"""Nyúl & Udupa piecewise-linear histogram matching normalization."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from intensity_normalization import _image
from intensity_normalization._image import ImageLike
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods._transform import FittedTransform

__all__ = ["NyulTransform", "fit", "fit_transform"]


def _percentile_grid(
    min_percentile: float,
    percentile_after_min: float,
    percentile_before_max: float,
    max_percentile: float,
    percentile_step: float,
) -> npt.NDArray[np.floating]:
    percs = np.arange(
        percentile_after_min,
        percentile_before_max + percentile_step,
        percentile_step,
    )
    return np.concatenate([[min_percentile], percs, [max_percentile]])


class NyulTransform(FittedTransform):
    """Piecewise-linear histogram matching learned from a set of images.

    Maps each image's landmark percentiles onto the population's standard
    scale. Attributes are the learned parameters, exposed read-only for
    inspection and plotting.
    """

    method: typing.ClassVar[str] = "nyul"

    def __init__(
        self,
        standard_scale: npt.NDArray[np.floating],
        percentiles: npt.NDArray[np.floating],
    ) -> None:
        self.standard_scale = np.asarray(standard_scale, dtype=np.float32)
        self.percentiles = np.asarray(percentiles, dtype=np.float32)

    def landmarks(self, intensities: npt.NDArray[np.floating], /) -> npt.NDArray[np.floating]:
        """Landmark intensities of a 1D foreground array."""
        return np.percentile(intensities, self.percentiles)

    def transform(self, image: ImageLike, /, mask: ImageLike | None = None) -> ImageLike:
        data, restore = _image.unwrap(image)
        mask_data = _image.unwrap_mask(image, mask)
        foreground = _image.foreground_values(data, mask_data)
        landmarks = self.landmarks(foreground)
        mapping = interp1d(
            landmarks,
            self.standard_scale,
            fill_value="extrapolate",
        )
        return restore(mapping(data).astype(np.float32))

    def _state_dict(self) -> dict[str, np.ndarray]:
        return {
            "standard_scale": self.standard_scale,
            "percentiles": self.percentiles,
        }

    @classmethod
    def _from_state_dict(cls, state: dict[str, np.ndarray]) -> NyulTransform:
        return cls(state["standard_scale"], state["percentiles"])


def fit(
    images: Sequence[ImageLike],
    /,
    masks: Sequence[ImageLike | None] | None = None,
    *,
    output_min_value: float = 1.0,
    output_max_value: float = 100.0,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
    percentile_after_min: float = 10.0,
    percentile_before_max: float = 90.0,
    percentile_step: float = 10.0,
) -> NyulTransform:
    """Learn the standard histogram scale from a population of images.

    Streams one image at a time, keeping only per-image landmark percentiles —
    the dataset size never bounds memory.

    Args:
        images: MR images (numpy arrays or nibabel images), all one modality.
        masks: optional foreground (brain) mask per image; where omitted, the
            foreground is estimated as positive voxels.
        output_min_value: intensity the ``min_percentile`` landmark maps to.
        output_max_value: intensity the ``max_percentile`` landmark maps to.
        min_percentile: lower percentile bound of the standard histogram.
        max_percentile: upper percentile bound of the standard histogram.
        percentile_after_min: first intermediate landmark percentile.
        percentile_before_max: last intermediate landmark percentile.
        percentile_step: step between intermediate landmark percentiles.

    Returns:
        A fitted :class:`NyulTransform`.
    """
    if len(images) == 0:
        raise IntensityNormalizationError("No images provided to fit.")
    if masks is not None and len(masks) != len(images):
        raise ValueError(f"Got {len(images)} images but {len(masks)} masks.")

    percentiles = _percentile_grid(
        min_percentile,
        percentile_after_min,
        percentile_before_max,
        max_percentile,
        percentile_step,
    )
    if np.any(percentiles <= 0.0) or np.any(percentiles >= 100.0) or np.any(np.diff(percentiles) <= 0.0):
        raise ValueError(
            f"Percentile configuration must be strictly increasing within (0, 100); got grid {percentiles.tolist()}."
        )
    standard_scale = np.zeros(len(percentiles))
    for i, image in enumerate(images):
        data, _ = _image.unwrap(image)
        mask = masks[i] if masks is not None else None
        mask_data = _image.unwrap_mask(image, mask)
        foreground = _image.foreground_values(data, mask_data)
        landmarks = np.percentile(foreground, percentiles)
        lo, hi = landmarks[0], landmarks[-1]
        if hi == lo:
            msg = (
                f"Image {i} has a degenerate foreground (the {min_percentile}th and "
                f"{max_percentile}th percentiles are both {lo:.3f}). Check its mask."
            )
            raise IntensityNormalizationError(msg)
        to_output = interp1d([lo, hi], [output_min_value, output_max_value])
        standard_scale += to_output(landmarks)
    standard_scale /= len(images)
    return NyulTransform(standard_scale, percentiles)


def fit_transform(
    images: Sequence[ImageLike],
    /,
    masks: Sequence[ImageLike | None] | None = None,
    **kwargs: typing.Any,
) -> tuple[NyulTransform, list[ImageLike]]:
    """Fit on ``images`` and return the transform plus the normalized images."""
    tx = fit(images, masks, **kwargs)
    normed = [tx(img, masks[i] if masks is not None else None) for i, img in enumerate(images)]
    return tx, normed
