"""Kernel density estimation-based tissue mode normalization."""

import numpy as np
import numpy.typing as npt
from scipy.signal import argrelmax
from scipy.stats import gaussian_kde

from intensity_normalization.domain.exceptions import NormalizationError
from intensity_normalization.domain.models import Modality
from intensity_normalization.domain.protocols import BaseNormalizer, ImageProtocol


class KDENormalizer(BaseNormalizer):
    """Kernel density estimation-based tissue mode normalization."""

    def __init__(self, modality: Modality = Modality.T1) -> None:
        super().__init__()
        self.modality = modality
        self._tissue_mode: float | None = None

    def fit(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> "KDENormalizer":
        """Fit KDE to find tissue mode."""
        data = image.get_data()

        if mask is not None:
            mask_data = mask.get_data().astype(bool)
            foreground = data[mask_data]
        else:
            foreground = data[data > 0]

        if len(foreground) == 0:
            raise NormalizationError("No foreground voxels found")

        try:
            self._tissue_mode = self._get_tissue_mode(foreground)
        except Exception as e:
            raise NormalizationError(f"KDE tissue mode calculation failed: {e}") from e

        self.is_fitted = True
        return self

    def transform(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Apply KDE normalization."""
        if not self.is_fitted or self._tissue_mode is None:
            raise NormalizationError("KDENormalizer must be fitted before transform")

        data = image.get_data()
        normalized_data = data / self._tissue_mode
        return image.with_data(normalized_data)

    def _get_tissue_mode(self, image_data: npt.NDArray[np.floating]) -> float:
        """Get tissue mode using KDE based on modality."""
        if self.modality == Modality.T1:
            return self._get_last_tissue_mode(image_data)
        if self.modality in (Modality.T2, Modality.FLAIR):
            return self._get_largest_tissue_mode(image_data)
        if self.modality == Modality.PD:
            return self._get_first_tissue_mode(image_data)
        return self._get_last_tissue_mode(image_data)

    def _smooth_histogram(
        self, image_data: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Use kernel density estimate to get smooth histogram."""
        image_vec = image_data.flatten().astype(np.float64)

        # Create KDE
        kde = gaussian_kde(image_vec)

        # Create grid for evaluation
        grid = np.linspace(image_vec.min(), image_vec.max(), 80)
        pdf = kde(grid)

        return grid, pdf

    def _get_largest_tissue_mode(self, image_data: npt.NDArray[np.floating]) -> float:
        """Mode of the largest tissue class."""
        grid, pdf = self._smooth_histogram(image_data)
        return float(grid[np.argmax(pdf)])

    def _get_last_tissue_mode(self, image_data: npt.NDArray[np.floating], tail_percentage: float = 96.0) -> float:
        """Mode of the highest-intensity tissue class."""
        # Remove tail
        threshold = float(np.percentile(image_data, tail_percentage))
        valid_data = image_data[image_data <= threshold]

        grid, pdf = self._smooth_histogram(valid_data)
        maxima = argrelmax(pdf)[0]

        if len(maxima) == 0:
            # If no maxima found, use the global maximum
            return float(grid[np.argmax(pdf)])

        return float(grid[maxima[-1]])

    def _get_first_tissue_mode(self, image_data: npt.NDArray[np.floating], tail_percentage: float = 99.0) -> float:
        """Mode of the lowest-intensity tissue class."""

        # Remove tail
        threshold = float(np.percentile(image_data, tail_percentage))
        valid_data = image_data[image_data <= threshold]

        grid, pdf = self._smooth_histogram(valid_data)
        maxima = argrelmax(pdf)[0]

        if len(maxima) == 0:
            # If no maxima found, use the global maximum
            return float(grid[np.argmax(pdf)])

        return float(grid[maxima[0]])

    @property
    def tissue_mode(self) -> float | None:
        """Get the fitted tissue mode."""
        return self._tissue_mode
