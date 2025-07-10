"""Nyul & Udupa piecewise linear histogram matching normalization."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from intensity_normalization.domain.exceptions import NormalizationError
from intensity_normalization.domain.protocols import ImageProtocol, PopulationNormalizer


class NyulNormalizer(PopulationNormalizer):
    """Nyul & Udupa piecewise linear histogram matching normalization."""

    def __init__(
        self,
        output_min_value: float = 1.0,
        output_max_value: float = 100.0,
        min_percentile: float = 1.0,
        max_percentile: float = 99.0,
        percentile_after_min: float = 10.0,
        percentile_before_max: float = 90.0,
        percentile_step: float = 10.0,
    ) -> None:
        super().__init__()
        self.output_min_value = output_min_value
        self.output_max_value = output_max_value
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.percentile_after_min = percentile_after_min
        self.percentile_before_max = percentile_before_max
        self.percentile_step = percentile_step
        self._percentiles: npt.NDArray[np.floating] | None = None
        self._standard_scale: npt.NDArray[np.floating] | None = None

    def fit_population(
        self,
        images: Sequence[ImageProtocol],
        masks: Sequence[ImageProtocol | None] | None = None,
    ) -> "NyulNormalizer":
        """Fit standard scale for piecewise linear histogram matching."""
        if len(images) == 0:
            raise NormalizationError("No images provided for fitting")

        if masks is not None and len(images) != len(masks):
            raise NormalizationError("Number of images and masks must match")

        n_percs = len(self.percentiles)
        standard_scale = np.zeros(n_percs)

        for i, image in enumerate(images):
            mask = masks[i] if masks else None
            voi = self._get_voi(image, mask)
            landmarks = self._get_landmarks(voi)

            # Scale landmarks to output range
            min_p = np.percentile(voi, self.min_percentile)
            max_p = np.percentile(voi, self.max_percentile)

            if min_p == max_p:
                raise NormalizationError(f"Min and max percentiles are equal in image {i}")

            f = interp1d(
                [min_p, max_p],
                [self.output_min_value, self.output_max_value],
                bounds_error=False,
                fill_value="extrapolate",
            )
            scaled_landmarks = f(landmarks)
            standard_scale += scaled_landmarks

        self._standard_scale = standard_scale / len(images)
        self.is_fitted = True
        return self

    def transform(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Apply Nyul normalization."""
        if not self.is_fitted or self._standard_scale is None:
            raise NormalizationError("NyulNormalizer must be fitted before transform")

        voi = self._get_voi(image, mask)
        landmarks = self._get_landmarks(voi)

        # Create interpolation function
        f = interp1d(
            landmarks,
            self._standard_scale,
            bounds_error=False,
            fill_value="extrapolate",
        )

        data = image.get_data()
        normalized_data = f(data)
        return image.with_data(normalized_data)

    def _get_voi(self, image: ImageProtocol, mask: ImageProtocol | None) -> npt.NDArray[np.floating]:
        """Get volume of interest (foreground voxels)."""
        data = image.get_data()

        if mask is not None:
            mask_data = mask.get_data().astype(bool)
            return data[mask_data]
        return data[data > 0]

    def _get_landmarks(self, voi: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get percentile landmarks for histogram matching."""
        return np.percentile(voi, self.percentiles)

    @property
    def percentiles(self) -> npt.NDArray[np.floating]:
        """Get the percentiles used for landmark computation."""
        if self._percentiles is None:
            percs = np.arange(
                self.percentile_after_min,
                self.percentile_before_max + self.percentile_step,
                self.percentile_step,
            )
            self._percentiles = np.concatenate(
                [
                    [self.min_percentile],
                    percs,
                    [self.max_percentile],
                ]
            )
        return self._percentiles

    @property
    def standard_scale(self) -> npt.NDArray[np.floating] | None:
        """Get the fitted standard scale."""
        return self._standard_scale

    def save_standard_histogram(self, filename: str) -> None:
        """Save standard histogram to file."""
        if self._standard_scale is None:
            raise NormalizationError("Normalizer must be fitted before saving")

        np.save(filename, np.vstack([self._standard_scale, self.percentiles]))

    def load_standard_histogram(self, filename: str) -> None:
        """Load standard histogram from file."""
        data = np.load(filename)
        self._standard_scale = data[0, :]
        self._percentiles = data[1, :]
        self.is_fitted = True
