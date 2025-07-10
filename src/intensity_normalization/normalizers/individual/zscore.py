"""Standard z-score intensity normalization."""

import numpy as np

from intensity_normalization.domain.exceptions import NormalizationError
from intensity_normalization.domain.protocols import BaseNormalizer, ImageProtocol


class ZScoreNormalizer(BaseNormalizer):
    """Standard z-score intensity normalization."""

    def __init__(self) -> None:
        super().__init__()
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> "ZScoreNormalizer":
        """Compute mean and std for z-score normalization."""
        data = image.get_data()

        if mask is not None:
            mask_data = mask.get_data().astype(bool)
            foreground = data[mask_data]
        else:
            foreground = data[data > 0]

        if len(foreground) == 0:
            raise NormalizationError("No foreground voxels found")

        self._mean = float(np.mean(foreground))
        self._std = float(np.std(foreground))

        if self._std == 0:
            raise NormalizationError("Standard deviation is zero - cannot normalize")

        self.is_fitted = True
        return self

    def transform(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Apply z-score normalization."""
        if not self.is_fitted or self._mean is None or self._std is None:
            raise NormalizationError("ZScoreNormalizer must be fitted before transform")

        data = image.get_data()
        normalized_data = (data - self._mean) / self._std
        return image.with_data(normalized_data)

    @property
    def mean(self) -> float | None:
        """Get the fitted mean."""
        return self._mean

    @property
    def std(self) -> float | None:
        """Get the fitted standard deviation."""
        return self._std
