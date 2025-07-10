"""Fuzzy C-means tissue-based intensity normalization."""

import numpy as np
import numpy.typing as npt
from skfuzzy import cmeans

from intensity_normalization.domain.exceptions import NormalizationError
from intensity_normalization.domain.models import TissueType
from intensity_normalization.domain.protocols import BaseNormalizer, ImageProtocol


class FCMNormalizer(BaseNormalizer):
    """Fuzzy C-means tissue-based intensity normalization."""

    def __init__(
        self,
        n_clusters: int = 3,
        tissue_type: TissueType = TissueType.WM,
        max_iter: int = 50,
        error_threshold: float = 0.005,
        fuzziness: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.tissue_type = tissue_type
        self.max_iter = max_iter
        self.error_threshold = error_threshold
        self.fuzziness = fuzziness
        self._tissue_mean: float | None = None
        self._tissue_membership: npt.NDArray[np.floating] | None = None

    def fit(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> "FCMNormalizer":
        """Fit FCM clustering to determine tissue means."""
        data = image.get_data()

        if mask is not None:
            mask_data = mask.get_data().astype(bool)
            foreground = data[mask_data]
        else:
            mask_data = None
            foreground = data[data > 0]

        if len(foreground) == 0:
            raise NormalizationError("No foreground voxels found")

        # Run FCM clustering
        try:
            centers, memberships, *_ = cmeans(
                foreground.reshape(1, -1),
                self.n_clusters,
                self.fuzziness,
                error=self.error_threshold,
                maxiter=self.max_iter,
            )
        except Exception as e:
            raise NormalizationError(f"FCM clustering failed: {e}") from e

        # Sort centers and memberships by intensity (CSF, GM, WM for T1)
        sorted_indices = np.argsort(centers.flatten())
        sorted_centers = centers.flatten()[sorted_indices]
        sorted_memberships = memberships[sorted_indices]

        # Select tissue mean based on tissue type
        self._tissue_mean = self._select_tissue_mean(sorted_centers)

        # Create tissue membership mask for the whole image
        tissue_membership = np.zeros(data.shape)
        tissue_idx = self._get_tissue_index()

        if mask_data is not None:
            tissue_membership[mask_data] = sorted_memberships[tissue_idx]
        else:
            fg_mask = data > 0
            tissue_membership[fg_mask] = sorted_memberships[tissue_idx]

        self._tissue_membership = tissue_membership
        self.is_fitted = True
        return self

    def transform(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Apply FCM normalization."""
        if not self.is_fitted or self._tissue_mean is None:
            raise NormalizationError("FCMNormalizer must be fitted before transform")

        data = image.get_data()

        if self._tissue_membership is not None:
            # Use tissue membership for weighted normalization
            tissue_mean = float(np.average(data, weights=self._tissue_membership))
        else:
            # Fall back to simple division by fitted tissue mean
            tissue_mean = self._tissue_mean

        if tissue_mean == 0:
            raise NormalizationError("Tissue mean is zero - cannot normalize")

        normalized_data = data / tissue_mean
        return image.with_data(normalized_data)

    def _select_tissue_mean(self, centers: npt.NDArray[np.floating]) -> float:
        """Select appropriate tissue cluster based on tissue type."""
        if self.tissue_type == TissueType.CSF:
            return float(centers[0])  # Lowest intensity
        if self.tissue_type == TissueType.GM:
            return float(centers[len(centers) // 2])  # Middle intensity
        # WM
        return float(centers[-1])  # Highest intensity

    def _get_tissue_index(self) -> int:
        """Get the index of the target tissue type."""
        if self.tissue_type == TissueType.CSF:
            return 0
        if self.tissue_type == TissueType.GM:
            return 1
        # WM
        return 2

    @property
    def tissue_membership(self) -> npt.NDArray[np.floating] | None:
        """Get the tissue membership mask."""
        return self._tissue_membership
