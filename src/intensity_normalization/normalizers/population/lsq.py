"""Least-squares fit tissue means normalization."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from skfuzzy import cmeans

from intensity_normalization.domain.exceptions import NormalizationError
from intensity_normalization.domain.protocols import ImageProtocol, PopulationNormalizer


class LSQNormalizer(PopulationNormalizer):
    """Least-squares fit tissue means normalization."""

    def __init__(self, norm_value: float = 1.0) -> None:
        super().__init__()
        self.norm_value = norm_value
        self._standard_tissue_means: npt.NDArray[np.floating] | None = None

    def fit_population(
        self,
        images: Sequence[ImageProtocol],
        masks: Sequence[ImageProtocol | None] | None = None,
    ) -> "LSQNormalizer":
        """Fit standard tissue means using the first image."""
        if len(images) == 0:
            raise NormalizationError("No images provided for fitting")

        # Use first image to establish standard tissue means
        first_image = images[0]
        first_mask = masks[0] if masks else None

        try:
            tissue_membership = self._get_tissue_membership(first_image, first_mask)

            # Normalize by CSF mean first
            data = first_image.get_data()
            csf_mean = self._calculate_tissue_mean(data, tissue_membership, 0)  # CSF is index 0

            if csf_mean == 0:
                raise NormalizationError("CSF mean is zero")

            normalized_data = (data / csf_mean) * self.norm_value

            # Calculate standard tissue means from normalized image
            self._standard_tissue_means = self._get_tissue_means(normalized_data, tissue_membership)

        except Exception as e:
            raise NormalizationError(f"LSQ fitting failed: {e}") from e

        self.is_fitted = True
        return self

    def transform(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Apply LSQ normalization."""
        if not self.is_fitted or self._standard_tissue_means is None:
            raise NormalizationError("LSQNormalizer must be fitted before transform")

        try:
            tissue_membership = self._get_tissue_membership(image, mask)
            data = image.get_data()

            # Calculate current tissue means
            current_tissue_means = self._get_tissue_means(data, tissue_membership)

            # Calculate scaling factor using least squares
            scaling_factor = self._calculate_scaling_factor(current_tissue_means)

            if scaling_factor == 0:
                raise NormalizationError("Scaling factor is zero")

            normalized_data = data / scaling_factor
            return image.with_data(normalized_data)

        except Exception as e:
            raise NormalizationError(f"LSQ transform failed: {e}") from e

    def _get_tissue_membership(self, image: ImageProtocol, mask: ImageProtocol | None) -> npt.NDArray[np.floating]:
        """Get tissue membership using FCM clustering."""
        data = image.get_data()

        if mask is not None:
            mask_data = mask.get_data()
            # Check if mask is already tissue membership (4D)
            if mask_data.ndim == data.ndim + 1:
                return mask_data

            # Otherwise use as brain mask
            mask_bool = mask_data.astype(bool)
            foreground = data[mask_bool]
        else:
            mask_bool = data > 0
            foreground = data[mask_bool]

        if len(foreground) == 0:
            raise NormalizationError("No foreground voxels found")

        # Run FCM clustering (3 classes: CSF, GM, WM)
        centers, memberships, *_ = cmeans(
            foreground.reshape(1, -1),
            3,  # n_clusters
            2.0,  # fuzziness
            error=0.005,
            maxiter=50,
        )

        # Sort by intensity (CSF=0, GM=1, WM=2)
        sorted_indices = np.argsort(centers.flatten())
        sorted_memberships = memberships[sorted_indices]

        # Create tissue membership volume
        tissue_membership = np.zeros((*data.shape, 3))
        for i in range(3):
            tissue_membership[..., i][mask_bool] = sorted_memberships[i]

        return tissue_membership

    def _get_tissue_means(
        self,
        data: npt.NDArray[np.floating],
        tissue_membership: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Calculate weighted tissue means."""
        n_tissues = tissue_membership.shape[-1]
        tissue_means = []

        for i in range(n_tissues):
            weights = tissue_membership[..., i]
            if np.sum(weights) > 0:
                mean = np.average(data, weights=weights)
                tissue_means.append(mean)
            else:
                tissue_means.append(0.0)

        return np.array(tissue_means).reshape(-1, 1)

    def _calculate_tissue_mean(
        self,
        data: npt.NDArray[np.floating],
        tissue_membership: npt.NDArray[np.floating],
        tissue_idx: int,
    ) -> float:
        """Calculate mean for a specific tissue type."""
        weights = tissue_membership[..., tissue_idx]
        if np.sum(weights) > 0:
            return float(np.average(data, weights=weights))
        return 0.0

    def _calculate_scaling_factor(self, tissue_means: npt.NDArray[np.floating]) -> float:
        """Calculate scaling factor using least squares."""
        if self._standard_tissue_means is None:
            raise NormalizationError("Standard tissue means not set")

        numerator = float(tissue_means.T @ tissue_means)
        denominator = float(tissue_means.T @ self._standard_tissue_means)

        if denominator == 0:
            raise NormalizationError("Denominator is zero in scaling factor calculation")

        return numerator / denominator

    @property
    def standard_tissue_means(self) -> npt.NDArray[np.floating] | None:
        """Get the fitted standard tissue means."""
        return self._standard_tissue_means

    def save_standard_tissue_means(self, filename: str) -> None:
        """Save standard tissue means to file."""
        if self._standard_tissue_means is None:
            raise NormalizationError("Normalizer must be fitted before saving")

        np.save(filename, self._standard_tissue_means)

    def load_standard_tissue_means(self, filename: str) -> None:
        """Load standard tissue means from file."""
        self._standard_tissue_means = np.load(filename)
        self.is_fitted = True
