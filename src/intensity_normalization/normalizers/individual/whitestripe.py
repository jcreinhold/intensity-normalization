"""WhiteStripe (normal-appearing white matter) normalization."""

import numpy as np
import numpy.typing as npt
from scipy.signal import argrelmax
from scipy.stats import gaussian_kde

from intensity_normalization.domain.exceptions import NormalizationError
from intensity_normalization.domain.models import Modality
from intensity_normalization.domain.protocols import BaseNormalizer, ImageProtocol


class WhiteStripeNormalizer(BaseNormalizer):
    """WhiteStripe (normal-appearing white matter) normalization."""

    def __init__(
        self,
        width: float = 0.05,
        width_l: float | None = None,
        width_u: float | None = None,
        modality: Modality = Modality.T1,
    ) -> None:
        super().__init__()
        self.width_l = width_l or width
        self.width_u = width_u or width
        self.modality = modality
        self._whitestripe_mask: npt.NDArray[np.bool_] | None = None
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> "WhiteStripeNormalizer":
        """Fit WhiteStripe to find normal-appearing white matter."""
        data = image.get_data()

        if mask is not None:
            mask_data = mask.get_data().astype(bool)
            foreground = data[mask_data]
        else:
            mask_data = data > 0
            foreground = data[mask_data]

        if len(foreground) == 0:
            raise NormalizationError("No foreground voxels found")

        try:
            # Get white matter mode
            wm_mode = self._get_tissue_mode(foreground)

            # Calculate quantile position of WM mode
            wm_mode_quantile = float(np.mean(foreground < wm_mode))

            # Calculate bounds
            lower_bound = max(wm_mode_quantile - self.width_l, 0.0)
            upper_bound = min(wm_mode_quantile + self.width_u, 1.0)

            # Get intensity thresholds
            ws_l, ws_u = np.quantile(foreground, (lower_bound, upper_bound))

            # Create whitestripe mask
            if mask is not None:
                masked_data = data * mask_data
                self._whitestripe_mask = (masked_data > ws_l) & (masked_data < ws_u)
            else:
                self._whitestripe_mask = (data > ws_l) & (data < ws_u)

            # Calculate mean and std of whitestripe region
            whitestripe_values = data[self._whitestripe_mask]

            if len(whitestripe_values) == 0:
                raise NormalizationError("No voxels found in white stripe region")

            self._mean = float(np.mean(whitestripe_values))
            self._std = float(np.std(whitestripe_values))

            if self._std == 0:
                raise NormalizationError("Standard deviation of white stripe is zero")

        except Exception as e:
            raise NormalizationError(f"WhiteStripe fitting failed: {e}") from e

        self.is_fitted = True
        return self

    def transform(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Apply WhiteStripe normalization."""
        if not self.is_fitted or self._mean is None or self._std is None:
            raise NormalizationError("WhiteStripeNormalizer must be fitted before transform")

        data = image.get_data()
        normalized_data = (data - self._mean) / self._std
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
        kde = gaussian_kde(image_vec)
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
            return float(grid[np.argmax(pdf)])

        return float(grid[maxima[0]])

    @property
    def whitestripe_mask(self) -> npt.NDArray[np.bool_] | None:
        """Get the white stripe mask."""
        return self._whitestripe_mask

    @property
    def mean(self) -> float | None:
        """Get the fitted mean."""
        return self._mean

    @property
    def std(self) -> float | None:
        """Get the fitted standard deviation."""
        return self._std
