"""Nyúl & Udupa piecewise-linear histogram matching normalization."""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Sequence

import numpy as np
from scipy.interpolate import interp1d

from intensity_normalization import _image
from intensity_normalization._image import BinaryMask, Image, IntensityArray, Mask
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods._transform import FittedTransform

__all__ = ["NyulTransform", "fit", "fit_array", "fit_transform"]

#: default landmark percentiles (Nyúl & Udupa 1998; Shah et al. 2011)
_DEFAULT_LANDMARKS = (1.0, *range(10, 91, 10), 99.0)


def _validate_landmarks(landmarks: Sequence[float] | None) -> IntensityArray:
    """The percentile grid, validated once: strictly increasing within (0, 100)."""
    grid = np.asarray(landmarks if landmarks is not None else _DEFAULT_LANDMARKS, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError(f"landmarks must be a 1D sequence of at least 3 percentiles; got {grid!r}.")
    if np.any(grid <= 0.0) or np.any(grid >= 100.0) or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"landmarks must be strictly increasing within (0, 100); got {grid.tolist()}.")
    return grid


@dataclasses.dataclass(frozen=True, eq=False)
class NyulTransform(FittedTransform):
    """Piecewise-linear histogram matching learned from a set of images.

    Maps each image's landmark percentiles onto the population's standard
    scale. Frozen and write-protected: the learned parameters cannot change
    after construction, so a saved transform always matches the in-memory one.
    """

    standard_scale: IntensityArray
    landmark_percentiles: IntensityArray

    method: typing.ClassVar[str] = "nyul"

    def __post_init__(self) -> None:
        object.__setattr__(self, "standard_scale", np.asarray(self.standard_scale, dtype=np.float32))
        object.__setattr__(self, "landmark_percentiles", np.asarray(self.landmark_percentiles, dtype=np.float32))
        self.standard_scale.setflags(write=False)
        self.landmark_percentiles.setflags(write=False)

    def landmark_intensities(self, intensities: IntensityArray) -> IntensityArray:
        """Landmark intensities of a 1D foreground array."""
        return np.percentile(intensities, self.landmark_percentiles)

    def transform_array(self, data: IntensityArray, foreground: BinaryMask, **kwargs: typing.Any) -> IntensityArray:
        foreground_values = _image.foreground_values(data, foreground)
        mapping = interp1d(
            self.landmark_intensities(foreground_values),
            self.standard_scale,
            fill_value="extrapolate",
        )
        return mapping(data).astype(np.float32)

    def _state_dict(self) -> dict[str, np.ndarray]:
        return {
            "standard_scale": self.standard_scale,
            "landmark_percentiles": self.landmark_percentiles,
        }

    @classmethod
    def _from_state_dict(cls, state: dict[str, np.ndarray]) -> NyulTransform:
        return cls(state["standard_scale"], state["landmark_percentiles"])


def fit_array(
    datas: Sequence[IntensityArray],
    foregrounds: Sequence[BinaryMask],
    *,
    landmarks: Sequence[float] | None = None,
    output_min_value: float = 1.0,
    output_max_value: float = 100.0,
) -> NyulTransform:
    """Learn the standard histogram scale from a population of intensity arrays.

    Streams one array at a time, keeping only per-array landmark percentiles —
    the dataset size never bounds memory.

    Args:
        datas: intensity arrays, all one modality.
        foregrounds: boolean foreground (brain) mask per array.
        landmarks: landmark percentiles, strictly increasing within (0, 100);
            defaults to the standard grid 1, 10, ..., 90, 99.
        output_min_value: intensity the first landmark maps to.
        output_max_value: intensity the last landmark maps to.

    Returns:
        A fitted :class:`NyulTransform`.
    """
    if len(datas) == 0:
        raise IntensityNormalizationError("No images provided to fit.")
    if len(foregrounds) != len(datas):
        raise ValueError(f"Got {len(datas)} images but {len(foregrounds)} foreground masks.")

    percentiles = _validate_landmarks(landmarks)
    standard_scale = np.zeros(len(percentiles))
    for i, (data, foreground) in enumerate(zip(datas, foregrounds, strict=True)):
        intensities = np.percentile(_image.foreground_values(data, foreground), percentiles)
        lo, hi = intensities[0], intensities[-1]
        if hi == lo:
            msg = (
                f"Image {i} has a degenerate foreground (the {percentiles[0]:g}th and "
                f"{percentiles[-1]:g}th percentiles are both {lo:.3f}). Check its mask."
            )
            raise IntensityNormalizationError(msg)
        to_output = interp1d([lo, hi], [output_min_value, output_max_value])
        standard_scale += to_output(intensities)
    standard_scale /= len(datas)
    return NyulTransform(standard_scale, percentiles)


def fit(
    images: Sequence[Image],
    masks: Sequence[Mask | None] | None = None,
    *,
    landmarks: Sequence[float] | None = None,
    output_min_value: float = 1.0,
    output_max_value: float = 100.0,
) -> NyulTransform:
    """Learn the standard histogram scale from a population of images; see :func:`fit_array`.

    Args:
        images: MR images (numpy arrays or nibabel images), all one modality.
        masks: optional foreground (brain) mask per image; where omitted, the
            foreground is estimated as positive voxels.
        landmarks: landmark percentiles, strictly increasing within (0, 100);
            defaults to the standard grid 1, 10, ..., 90, 99.
        output_min_value: intensity the first landmark maps to.
        output_max_value: intensity the last landmark maps to.

    Returns:
        A fitted :class:`NyulTransform`.
    """
    if len(images) == 0:
        raise IntensityNormalizationError("No images provided to fit.")
    if masks is not None and len(masks) != len(images):
        raise ValueError(f"Got {len(images)} images but {len(masks)} masks.")
    datas = [_image.unwrap(image)[0] for image in images]
    foregrounds = [
        _image.resolve_foreground(data, _image.unwrap_mask(image, masks[i] if masks is not None else None))
        for i, (image, data) in enumerate(zip(images, datas, strict=True))
    ]
    return fit_array(
        datas,
        foregrounds,
        landmarks=landmarks,
        output_min_value=output_min_value,
        output_max_value=output_max_value,
    )


def fit_transform(
    images: Sequence[Image],
    masks: Sequence[Mask | None] | None = None,
    **kwargs: typing.Any,
) -> tuple[NyulTransform, list[Image]]:
    """Fit on ``images`` and return the transform plus the normalized images."""
    tx = fit(images, masks, **kwargs)
    normed = [tx(img, masks[i] if masks is not None else None) for i, img in enumerate(images)]
    return tx, normed
