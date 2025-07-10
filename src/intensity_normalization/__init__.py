"""Top-level package for intensity-normalization."""

import importlib.metadata
import logging

# Modern API exports
from intensity_normalization.adapters.images import ImageProtocol, create_image
from intensity_normalization.domain.models import Modality, NormalizationConfig, TissueType
from intensity_normalization.normalizers.individual.fcm import FCMNormalizer
from intensity_normalization.normalizers.individual.kde import KDENormalizer
from intensity_normalization.normalizers.individual.whitestripe import WhiteStripeNormalizer
from intensity_normalization.normalizers.individual.zscore import ZScoreNormalizer
from intensity_normalization.normalizers.population.lsq import LSQNormalizer
from intensity_normalization.normalizers.population.nyul import NyulNormalizer
from intensity_normalization.services.normalization import NormalizationService

# Module constants
__version__ = importlib.metadata.version("intensity-normalization")

# Legacy compatibility
PEAK = {
    "last": ("t1", "other", "last"),
    "largest": ("t2", "flair", "largest"),
    "first": ("pd", "md", "first"),
}
VALID_PEAKS = frozenset({m for modalities in PEAK.values() for m in modalities})
VALID_MODALITIES = VALID_PEAKS - {"last", "largest", "first"}


def normalize_image(
    image,
    method: str = "fcm",
    mask=None,
    modality: str = "t1",
    tissue_type: str = "wm",
) -> ImageProtocol:
    """High-level function to normalize a single image.

    Args:
        image: Image to normalize (numpy array, nibabel image, or file path)
        method: Normalization method ("fcm", "zscore", "kde", "whitestripe")
        mask: Optional brain mask (same formats as image)
        modality: MR modality ("t1", "t2", "flair", "pd")
        tissue_type: Target tissue type ("wm", "gm", "csf")

    Returns:
        Normalized image (same type as input)
    """
    # Convert to image protocol
    image_obj = create_image(image)
    mask_obj = create_image(mask) if mask is not None else None

    # Create configuration
    config = NormalizationConfig(
        method=method,
        modality=Modality(modality),
        tissue_type=TissueType(tissue_type),
    )

    # Normalize
    return NormalizationService.normalize_image(image_obj, config, mask_obj)


__all__ = [  # noqa: RUF022
    # Main API
    "normalize_image",
    "create_image",
    # Configuration
    "NormalizationConfig",
    "Modality",
    "TissueType",
    # Normalizers
    "FCMNormalizer",
    "KDENormalizer",
    "LSQNormalizer",
    "NyulNormalizer",
    "WhiteStripeNormalizer",
    "ZScoreNormalizer",
    # Services
    "NormalizationService",
    # Legacy compatibility
    "PEAK",
    "VALID_PEAKS",
    "VALID_MODALITIES",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
