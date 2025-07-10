"""Input validation services."""

import inspect
import os
from collections.abc import Sequence
from pathlib import Path

from intensity_normalization.domain.exceptions import ValidationError
from intensity_normalization.domain.models import Modality, NormalizationConfig, TissueType
from intensity_normalization.domain.protocols import ImageProtocol
from intensity_normalization.services.normalization import NORMALIZER_REGISTRY


class ValidationService:
    """Service for validating inputs and configurations."""

    @staticmethod
    def validate_normalization_config(config: NormalizationConfig) -> None:
        """Validate normalization configuration."""
        # Validate method
        if config.method not in NORMALIZER_REGISTRY:
            available = ", ".join(sorted(NORMALIZER_REGISTRY.keys()))
            raise ValidationError(f"Invalid method '{config.method}'. Available: {available}")

        # Validate modality
        if not isinstance(config.modality, Modality):
            raise ValidationError(f"Invalid modality type: {type(config.modality)}")

        # Validate tissue type
        if not isinstance(config.tissue_type, TissueType):
            raise ValidationError(f"Invalid tissue type: {type(config.tissue_type)}")

    @staticmethod
    def validate_image_list(images: Sequence[ImageProtocol], name: str = "images") -> None:
        """Validate list of images."""
        if len(images) == 0:
            raise ValidationError(f"{name} list cannot be empty")

    @staticmethod
    def validate_mask_list(masks: Sequence[ImageProtocol] | None, num_images: int, name: str = "masks") -> None:
        """Validate list of masks."""
        if masks is None:
            return

        if len(masks) != num_images:
            raise ValidationError(f"Number of {name} ({len(masks)}) must match number of images ({num_images})")

    @staticmethod
    def validate_file_path(path: str | os.PathLike, must_exist: bool = True) -> None:
        """Validate file path."""
        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            raise ValidationError(f"File does not exist: {path_obj}")

        if must_exist and not path_obj.is_file():
            raise ValidationError(f"Path is not a file: {path_obj}")

    @staticmethod
    def validate_output_path(path: str | os.PathLike) -> None:
        """Validate output file path."""
        path_obj = Path(path)

        # Check if parent directory exists
        if not path_obj.parent.exists():
            raise ValidationError(f"Output directory does not exist: {path_obj.parent}")

        # Check if parent is actually a directory
        if not path_obj.parent.is_dir():
            raise ValidationError(f"Output parent is not a directory: {path_obj.parent}")

    @staticmethod
    def validate_method_parameters(method: str, **kwargs) -> None:
        """Validate method-specific parameters."""

        if method not in NORMALIZER_REGISTRY:
            available = ", ".join(sorted(NORMALIZER_REGISTRY.keys()))
            raise ValidationError(f"Invalid method '{method}'. Available: {available}")

        normalizer_cls = NORMALIZER_REGISTRY[method]

        # Get valid parameters from __init__ signature
        try:
            sig = inspect.signature(normalizer_cls)
            valid_params = set(sig.parameters.keys()) - {"self"}

            # Check for invalid parameters
            invalid_params = set(kwargs.keys()) - valid_params
            if invalid_params:
                invalid_list = ", ".join(sorted(invalid_params))
                valid_list = ", ".join(sorted(valid_params))
                raise ValidationError(
                    f"Invalid parameters for {method}: {invalid_list}. Valid parameters: {valid_list}"
                )

        except Exception:
            # If we can't inspect, just try to create the normalizer
            try:
                normalizer_cls(**kwargs)
            except Exception as creation_error:
                raise ValidationError(f"Invalid parameters for {method}: {creation_error}") from creation_error
