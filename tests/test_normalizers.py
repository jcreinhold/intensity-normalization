"""Test modern normalizer implementations."""

from __future__ import annotations

import pytest

from intensity_normalization.adapters.images import NumpyImageAdapter
from intensity_normalization.domain.exceptions import NormalizationError
from intensity_normalization.domain.models import TissueType
from intensity_normalization.domain.protocols import ImageProtocol
from intensity_normalization.normalizers.individual.fcm import FCMNormalizer
from intensity_normalization.normalizers.individual.kde import KDENormalizer
from intensity_normalization.normalizers.individual.whitestripe import WhiteStripeNormalizer
from intensity_normalization.normalizers.individual.zscore import ZScoreNormalizer
from intensity_normalization.normalizers.population.lsq import LSQNormalizer
from intensity_normalization.normalizers.population.nyul import NyulNormalizer


class TestZScoreNormalizer:
    """Test Z-score normalization."""

    def test_initialization(self) -> None:
        normalizer = ZScoreNormalizer()
        assert not normalizer.is_fitted
        assert normalizer.mean is None
        assert normalizer.std is None

    def test_fit_transform(self, image_fixture: ImageProtocol, mask_fixture: ImageProtocol) -> None:
        normalizer = ZScoreNormalizer()
        result = normalizer.fit_transform(image_fixture, mask_fixture)

        assert isinstance(result, type(image_fixture))
        assert result.shape == image_fixture.shape
        assert normalizer.is_fitted
        assert normalizer.mean is not None
        assert normalizer.std is not None

    def test_fit_transform_no_mask(self, image_fixture: ImageProtocol) -> None:
        normalizer = ZScoreNormalizer()
        result = normalizer.fit_transform(image_fixture)

        assert result.shape == image_fixture.shape
        assert normalizer.is_fitted

    def test_transform_before_fit_raises_error(self, image_fixture: ImageProtocol) -> None:
        normalizer = ZScoreNormalizer()
        with pytest.raises(NormalizationError, match="must be fitted"):
            normalizer.transform(image_fixture)

    def test_call_method(self, image_fixture: ImageProtocol) -> None:
        """Test __call__ method works as expected."""
        normalizer = ZScoreNormalizer()

        # First call should fit and transform
        result1 = normalizer(image_fixture)
        assert normalizer.is_fitted

        # Second call should just transform
        result2 = normalizer(image_fixture)
        assert result1.shape == result2.shape


class TestFCMNormalizer:
    """Test FCM normalization."""

    def test_initialization(self) -> None:
        normalizer = FCMNormalizer()
        assert normalizer.n_clusters == 3
        assert normalizer.tissue_type == TissueType.WM
        assert not normalizer.is_fitted

    def test_fit_transform(self, image_fixture: ImageProtocol, mask_fixture: ImageProtocol) -> None:
        normalizer = FCMNormalizer(tissue_type=TissueType.WM)
        result = normalizer.fit_transform(image_fixture, mask_fixture)

        assert result.shape == image_fixture.shape
        assert normalizer.is_fitted
        assert normalizer.tissue_membership is not None

    @pytest.mark.parametrize("tissue_type", list(TissueType))
    def test_different_tissue_types(self, image_fixture: ImageProtocol, tissue_type: TissueType) -> None:
        normalizer = FCMNormalizer(tissue_type=tissue_type)
        result = normalizer.fit_transform(image_fixture)
        assert result.shape == image_fixture.shape


class TestKDENormalizer:
    """Test KDE normalization."""

    def test_initialization(self) -> None:
        normalizer = KDENormalizer()
        assert not normalizer.is_fitted
        assert normalizer.tissue_mode is None

    def test_fit_transform(self, image_fixture: ImageProtocol) -> None:
        normalizer = KDENormalizer()
        result = normalizer.fit_transform(image_fixture)

        assert result.shape == image_fixture.shape
        assert normalizer.is_fitted
        assert normalizer.tissue_mode is not None


class TestWhiteStripeNormalizer:
    """Test WhiteStripe normalization."""

    def test_initialization(self) -> None:
        normalizer = WhiteStripeNormalizer()
        assert normalizer.width_l == 0.05
        assert normalizer.width_u == 0.05
        assert not normalizer.is_fitted

    def test_fit_transform(self, image_fixture: ImageProtocol) -> None:
        normalizer = WhiteStripeNormalizer(width=0.1)
        result = normalizer.fit_transform(image_fixture)

        assert result.shape == image_fixture.shape
        assert normalizer.is_fitted
        assert normalizer.whitestripe_mask is not None
        assert normalizer.mean is not None
        assert normalizer.std is not None


class TestNyulNormalizer:
    """Test Nyul normalization."""

    def test_initialization(self) -> None:
        normalizer = NyulNormalizer()
        assert normalizer.output_min_value == 1.0
        assert normalizer.output_max_value == 100.0
        assert not normalizer.is_fitted

    def test_fit_population(self, multiple_images: list[ImageProtocol]) -> None:
        normalizer = NyulNormalizer()
        normalizer.fit_population(multiple_images)

        assert normalizer.is_fitted
        assert normalizer.standard_scale is not None

    def test_fit_transform_multiple(
        self, multiple_images: list[ImageProtocol], multiple_masks: list[ImageProtocol]
    ) -> None:
        normalizer = NyulNormalizer()
        normalizer.fit_population(multiple_images, multiple_masks)

        results = []
        for i, image in enumerate(multiple_images):
            mask = multiple_masks[i]
            result = normalizer.transform(image, mask)
            results.append(result)

        assert len(results) == len(multiple_images)
        for result, original in zip(results, multiple_images, strict=False):
            assert result.shape == original.shape


class TestLSQNormalizer:
    """Test LSQ normalization."""

    def test_initialization(self) -> None:
        normalizer = LSQNormalizer()
        assert normalizer.norm_value == 1.0
        assert not normalizer.is_fitted

    def test_fit_population(self, multiple_images: list[ImageProtocol]) -> None:
        normalizer = LSQNormalizer()
        normalizer.fit_population(multiple_images)

        assert normalizer.is_fitted
        assert normalizer.standard_tissue_means is not None

    def test_fit_transform_multiple(self, multiple_images: list[ImageProtocol]) -> None:
        normalizer = LSQNormalizer()
        normalizer.fit_population(multiple_images)

        results = []
        for image in multiple_images:
            result = normalizer.transform(image)
            results.append(result)

        assert len(results) == len(multiple_images)
        for result, original in zip(results, multiple_images, strict=False):
            assert result.shape == original.shape


class TestNormalizersCompatibility:
    """Test that normalizers work with both numpy and nibabel images."""

    @pytest.mark.parametrize(
        "normalizer_cls",
        [
            ZScoreNormalizer,
            FCMNormalizer,
            KDENormalizer,
            WhiteStripeNormalizer,
        ],
    )
    def test_numpy_nibabel_compatibility(
        self,
        normalizer_cls,
        numpy_image: NumpyImageAdapter,
        nibabel_image,
    ) -> None:
        """Test that normalizers work with both numpy and nibabel images."""
        # Test with numpy image
        normalizer1 = normalizer_cls()
        result1 = normalizer1.fit_transform(numpy_image)
        assert isinstance(result1, NumpyImageAdapter)

        # Test with nibabel image
        normalizer2 = normalizer_cls()
        result2 = normalizer2.fit_transform(nibabel_image)
        assert hasattr(result2, "affine")  # NibabelImageAdapter should have affine


class TestErrorHandling:
    """Test error handling in normalizers."""

    def test_empty_foreground_error(self) -> None:
        """Test that normalizers handle empty foreground gracefully."""
        import numpy as np

        # Create image with all zeros
        empty_data = np.zeros((10, 10, 10), dtype=np.float32)
        empty_image = NumpyImageAdapter(empty_data)

        normalizer = ZScoreNormalizer()
        with pytest.raises(NormalizationError, match="No foreground voxels"):
            normalizer.fit(empty_image)

    def test_zero_std_error(self) -> None:
        """Test handling of zero standard deviation."""
        import numpy as np

        # Create image with constant values
        constant_data = np.ones((10, 10, 10), dtype=np.float32)
        constant_image = NumpyImageAdapter(constant_data)

        normalizer = ZScoreNormalizer()
        with pytest.raises(NormalizationError, match="Standard deviation is zero"):
            normalizer.fit(constant_image)
