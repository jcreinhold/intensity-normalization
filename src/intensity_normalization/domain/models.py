"""Domain models for intensity normalization."""

from dataclasses import dataclass
from enum import Enum


class Modality(Enum):
    """MR image modality types."""

    T1 = "t1"
    T2 = "t2"
    FLAIR = "flair"
    PD = "pd"
    OTHER = "other"


class TissueType(Enum):
    """Brain tissue types for normalization."""

    WM = "wm"  # White matter
    GM = "gm"  # Gray matter
    CSF = "csf"  # Cerebrospinal fluid


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for normalization methods."""

    method: str
    modality: Modality = Modality.T1
    tissue_type: TissueType = TissueType.WM


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata for medical images."""

    shape: tuple[int, ...]
    dtype: str
    spacing: tuple[float, ...] | None = None
    origin: tuple[float, ...] | None = None
