"""High-level normalization service for orchestrating normalization operations."""

import inspect
from collections.abc import Sequence
from typing import Any

from intensity_normalization.domain.exceptions import ConfigurationError, NormalizationError
from intensity_normalization.domain.models import NormalizationConfig
from intensity_normalization.domain.protocols import BaseNormalizer, ImageProtocol, PopulationNormalizer
from intensity_normalization.normalizers.individual.fcm import FCMNormalizer
from intensity_normalization.normalizers.individual.kde import KDENormalizer
from intensity_normalization.normalizers.individual.whitestripe import WhiteStripeNormalizer
from intensity_normalization.normalizers.individual.zscore import ZScoreNormalizer
from intensity_normalization.normalizers.population.lsq import LSQNormalizer
from intensity_normalization.normalizers.population.nyul import NyulNormalizer

# Registry of available normalizers
NORMALIZER_REGISTRY = {
    "fcm": FCMNormalizer,
    "zscore": ZScoreNormalizer,
    "kde": KDENormalizer,
    "whitestripe": WhiteStripeNormalizer,
    "nyul": NyulNormalizer,
    "lsq": LSQNormalizer,
}

# Population-based methods that need multiple images for fitting
POPULATION_METHODS = {"nyul", "lsq"}


class NormalizationService:
    """High-level service for image normalization orchestration."""

    @staticmethod
    def normalize_image(
        image: ImageProtocol,
        config: NormalizationConfig,
        mask: ImageProtocol | None = None,
    ) -> ImageProtocol:
        """Normalize a single image using the specified configuration."""
        normalizer = NormalizationService._create_normalizer(config)
        return normalizer.fit_transform(image, mask)

    @staticmethod
    def normalize_images(
        images: Sequence[ImageProtocol],
        config: NormalizationConfig,
        masks: Sequence[ImageProtocol | None] | None = None,
    ) -> list[ImageProtocol]:
        """Normalize multiple images with method-appropriate fitting."""
        if len(images) == 0:
            raise NormalizationError("No images provided for normalization")

        normalizer = NormalizationService._create_normalizer(config)

        # Validate masks if provided
        if masks is not None and len(images) != len(masks):
            raise NormalizationError("Number of images and masks must match")

        # Handle population vs individual methods
        if config.method in POPULATION_METHODS:
            return NormalizationService._normalize_with_population_method(normalizer, images, masks)
        return NormalizationService._normalize_with_individual_method(normalizer, images, masks)

    @staticmethod
    def _normalize_with_population_method(
        normalizer: BaseNormalizer,
        images: Sequence[ImageProtocol],
        masks: Sequence[ImageProtocol | None] | None,
    ) -> list[ImageProtocol]:
        """Normalize using population-based methods (fit once, transform all)."""
        if not isinstance(normalizer, PopulationNormalizer):
            raise ConfigurationError(f"Expected PopulationNormalizer, got {type(normalizer)}")

        # Fit on all images
        normalizer.fit_population(images, masks)

        # Transform each image
        normalized_images = []
        for i, image in enumerate(images):
            mask = masks[i] if masks else None
            normalized = normalizer.transform(image, mask)
            normalized_images.append(normalized)

        return normalized_images

    @staticmethod
    def _normalize_with_individual_method(
        normalizer: BaseNormalizer,
        images: Sequence[ImageProtocol],
        masks: Sequence[ImageProtocol | None] | None,
    ) -> list[ImageProtocol]:
        """Normalize using individual methods (fit each separately)."""
        normalized_images = []

        for i, image in enumerate(images):
            mask = masks[i] if masks else None
            # Create new normalizer instance for each image to avoid state conflicts
            image_normalizer = NormalizationService._create_normalizer_from_instance(normalizer)
            normalized = image_normalizer.fit_transform(image, mask)
            normalized_images.append(normalized)

        return normalized_images

    @staticmethod
    def create_normalizer(method: str, **kwargs) -> BaseNormalizer:
        """Factory method to create normalizer instances with custom parameters."""
        if method not in NORMALIZER_REGISTRY:
            available = ", ".join(sorted(NORMALIZER_REGISTRY.keys()))
            raise ConfigurationError(f"Unknown method '{method}'. Available: {available}")

        normalizer_cls = NORMALIZER_REGISTRY[method]

        # Filter kwargs to only include valid parameters for this normalizer
        try:
            return normalizer_cls(**kwargs)
        except TypeError as e:
            raise ConfigurationError(f"Invalid parameters for {method}: {e}") from e

    @staticmethod
    def _create_normalizer(config: NormalizationConfig) -> BaseNormalizer:
        """Create normalizer instance from configuration."""
        if config.method not in NORMALIZER_REGISTRY:
            available = ", ".join(sorted(NORMALIZER_REGISTRY.keys()))
            raise ConfigurationError(f"Unknown method '{config.method}'. Available: {available}")

        normalizer_cls = NORMALIZER_REGISTRY[config.method]

        # Pass configuration parameters to normalizer
        kwargs: dict[str, Any] = {}

        # Add tissue_type for methods that support it
        if hasattr(normalizer_cls, "__init__"):
            init_params = inspect.signature(normalizer_cls).parameters
            if "tissue_type" in init_params:
                kwargs["tissue_type"] = config.tissue_type
            if "modality" in init_params:
                kwargs["modality"] = config.modality

        try:
            return normalizer_cls(**kwargs)
        except Exception as e:
            raise ConfigurationError(f"Failed to create {config.method} normalizer: {e}") from e

    @staticmethod
    def _create_normalizer_from_instance(normalizer: BaseNormalizer) -> BaseNormalizer:
        """Create a new instance of the same normalizer type."""
        normalizer_cls = type(normalizer)

        # Try to preserve key parameters
        kwargs: dict[str, Any] = {}
        if hasattr(normalizer, "tissue_type"):
            kwargs["tissue_type"] = normalizer.tissue_type
        if hasattr(normalizer, "modality"):
            kwargs["modality"] = normalizer.modality
        if hasattr(normalizer, "width"):
            kwargs["width"] = normalizer.width
        if hasattr(normalizer, "n_clusters"):
            kwargs["n_clusters"] = normalizer.n_clusters

        return normalizer_cls(**kwargs)

    @staticmethod
    def get_available_methods() -> list[str]:
        """Get list of available normalization methods."""
        return sorted(NORMALIZER_REGISTRY.keys())

    @staticmethod
    def is_population_method(method: str) -> bool:
        """Check if a method requires multiple images for fitting."""
        return method in POPULATION_METHODS
