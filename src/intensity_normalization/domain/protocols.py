"""Domain protocols for intensity normalization."""

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class ImageProtocol(Protocol):
    """Protocol for medical images supporting both numpy arrays and nibabel images."""

    def get_data(self) -> npt.NDArray[np.floating]:
        """Get image data as numpy array."""
        ...

    def with_data(self, data: npt.NDArray[np.floating]) -> "ImageProtocol":
        """Create new image with updated data, preserving metadata."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get image shape."""
        ...

    def save(self, path: str | os.PathLike) -> None:
        """Save image to file."""
        ...


class BaseNormalizer(ABC):
    """Abstract base class for all normalization methods."""

    def __init__(self) -> None:
        self.is_fitted = False

    @abstractmethod
    def fit(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> "BaseNormalizer":
        """Fit normalization parameters to image(s)."""

    @abstractmethod
    def transform(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Apply normalization to image."""

    def fit_transform(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Fit and transform in one step."""
        return self.fit(image, mask).transform(image, mask)

    def __call__(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> ImageProtocol:
        """Convenience method - fit_transform for unfitted, transform for fitted."""
        if not self.is_fitted:
            return self.fit_transform(image, mask)
        return self.transform(image, mask)


class PopulationNormalizer(BaseNormalizer):
    """Base for normalizers that fit on multiple images."""

    @abstractmethod
    def fit_population(
        self,
        images: Sequence[ImageProtocol],
        masks: Sequence[ImageProtocol | None] | None = None,
    ) -> "PopulationNormalizer":
        """Fit on multiple images."""

    def fit(self, image: ImageProtocol, mask: ImageProtocol | None = None) -> "PopulationNormalizer":
        """Fit on a single image (delegates to fit_population)."""
        return self.fit_population([image], [mask] if mask else None)
