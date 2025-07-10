"""Domain exceptions for intensity normalization."""


class IntensityNormalizationError(Exception):
    """Base exception for intensity normalization errors."""


class ImageLoadError(IntensityNormalizationError):
    """Exception raised when image loading fails."""


class NormalizationError(IntensityNormalizationError):
    """Exception raised when normalization fails."""


class ValidationError(IntensityNormalizationError):
    """Exception raised when input validation fails."""


class ConfigurationError(IntensityNormalizationError):
    """Exception raised when configuration is invalid."""
