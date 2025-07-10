"""Test service layer functionality."""

import pytest

from intensity_normalization.domain.exceptions import ConfigurationError, NormalizationError, ValidationError
from intensity_normalization.domain.models import Modality, NormalizationConfig, TissueType
from intensity_normalization.domain.protocols import ImageProtocol
from intensity_normalization.services.normalization import NormalizationService
from intensity_normalization.services.validation import ValidationService


class TestNormalizationService:
    """Test the normalization service."""

    def test_normalize_single_image(self, image_fixture: ImageProtocol, mask_fixture: ImageProtocol) -> None:
        """Test normalizing a single image."""
        config = NormalizationConfig(
            method="zscore",
            modality=Modality.T1,
            tissue_type=TissueType.WM,
        )

        result = NormalizationService.normalize_image(image_fixture, config, mask_fixture)

        assert result.shape == image_fixture.shape
        assert result is not image_fixture  # Should be a new instance

    def test_normalize_multiple_images_individual_method(
        self, multiple_images: list[ImageProtocol], multiple_masks: list[ImageProtocol]
    ) -> None:
        """Test normalizing multiple images with individual method."""
        config = NormalizationConfig(method="zscore")

        results = NormalizationService.normalize_images(multiple_images, config, multiple_masks)

        assert len(results) == len(multiple_images)
        for result, original in zip(results, multiple_images, strict=False):
            assert result.shape == original.shape

    def test_normalize_multiple_images_population_method(
        self, multiple_images: list[ImageProtocol], multiple_masks: list[ImageProtocol]
    ) -> None:
        """Test normalizing multiple images with population method."""
        config = NormalizationConfig(method="nyul")

        results = NormalizationService.normalize_images(multiple_images, config, multiple_masks)

        assert len(results) == len(multiple_images)
        for result, original in zip(results, multiple_images, strict=False):
            assert result.shape == original.shape

    def test_create_normalizer(self) -> None:
        """Test normalizer factory method."""
        normalizer = NormalizationService.create_normalizer("zscore")
        assert normalizer is not None

        # Test with invalid method
        with pytest.raises(ConfigurationError, match="Unknown method"):
            NormalizationService.create_normalizer("invalid_method")

    def test_get_available_methods(self) -> None:
        """Test getting available methods."""
        methods = NormalizationService.get_available_methods()

        assert isinstance(methods, list)
        assert "zscore" in methods
        assert "fcm" in methods
        assert "nyul" in methods
        assert "lsq" in methods

    def test_is_population_method(self) -> None:
        """Test checking if method is population-based."""
        assert not NormalizationService.is_population_method("zscore")
        assert not NormalizationService.is_population_method("fcm")
        assert NormalizationService.is_population_method("nyul")
        assert NormalizationService.is_population_method("lsq")

    def test_empty_images_error(self) -> None:
        """Test error handling for empty image list."""
        config = NormalizationConfig(method="zscore")

        with pytest.raises(NormalizationError, match="No images provided"):
            NormalizationService.normalize_images([], config)

    def test_mismatched_masks_error(self, multiple_images: list[ImageProtocol]) -> None:
        """Test error handling for mismatched masks."""
        config = NormalizationConfig(method="zscore")
        wrong_masks = [None]  # Wrong number of masks

        with pytest.raises(NormalizationError, match="Number of images and masks must match"):
            NormalizationService.normalize_images(multiple_images, config, wrong_masks)


class TestValidationService:
    """Test the validation service."""

    def test_validate_normalization_config_valid(self) -> None:
        """Test validation of valid configuration."""
        config = NormalizationConfig(
            method="zscore",
            modality=Modality.T1,
            tissue_type=TissueType.WM,
        )

        # Should not raise
        ValidationService.validate_normalization_config(config)

    def test_validate_normalization_config_invalid_method(self) -> None:
        """Test validation with invalid method."""
        config = NormalizationConfig(
            method="invalid_method",
            modality=Modality.T1,
            tissue_type=TissueType.WM,
        )

        with pytest.raises(ValidationError, match="Invalid method"):
            ValidationService.validate_normalization_config(config)

    def test_validate_image_list_valid(self, multiple_images: list[ImageProtocol]) -> None:
        """Test validation of valid image list."""
        # Should not raise
        ValidationService.validate_image_list(multiple_images)

    def test_validate_image_list_empty(self) -> None:
        """Test validation of empty image list."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            ValidationService.validate_image_list([])

    def test_validate_mask_list_valid(self, multiple_masks: list[ImageProtocol]) -> None:
        """Test validation of valid mask list."""
        # Should not raise
        ValidationService.validate_mask_list(multiple_masks, len(multiple_masks))

    def test_validate_mask_list_none(self) -> None:
        """Test validation of None masks."""
        # Should not raise
        ValidationService.validate_mask_list(None, 3)

    def test_validate_mask_list_wrong_length(self, multiple_masks: list[ImageProtocol]) -> None:
        """Test validation of wrong mask list length."""
        with pytest.raises(ValidationError, match="must match number of images"):
            ValidationService.validate_mask_list(multiple_masks, len(multiple_masks) + 1)

    def test_validate_method_parameters(self) -> None:
        """Test validation of method parameters."""
        # Valid parameters
        ValidationService.validate_method_parameters("zscore")
        ValidationService.validate_method_parameters("fcm", tissue_type=TissueType.WM)

        # Invalid method
        with pytest.raises(ValidationError, match="Invalid method"):
            ValidationService.validate_method_parameters("invalid_method")


class TestIntegration:
    """Integration tests for services."""

    def test_end_to_end_normalization(self, image_fixture: ImageProtocol) -> None:
        """Test complete normalization workflow."""
        # Create configuration
        config = NormalizationConfig(
            method="zscore",
            modality=Modality.T1,
            tissue_type=TissueType.WM,
        )

        # Validate configuration
        ValidationService.validate_normalization_config(config)

        # Normalize image
        result = NormalizationService.normalize_image(image_fixture, config)

        # Verify result
        assert result.shape == image_fixture.shape
        assert result is not image_fixture

    @pytest.mark.parametrize("method", ["zscore", "fcm", "kde", "whitestripe"])
    def test_all_individual_methods(self, method: str, image_fixture: ImageProtocol) -> None:
        """Test all individual normalization methods."""
        config = NormalizationConfig(method=method)
        result = NormalizationService.normalize_image(image_fixture, config)
        assert result.shape == image_fixture.shape

    @pytest.mark.parametrize("method", ["nyul", "lsq"])
    def test_all_population_methods(self, method: str, multiple_images: list[ImageProtocol]) -> None:
        """Test all population normalization methods."""
        config = NormalizationConfig(method=method)
        results = NormalizationService.normalize_images(multiple_images, config)
        assert len(results) == len(multiple_images)
        for result, original in zip(results, multiple_images, strict=False):
            assert result.shape == original.shape
