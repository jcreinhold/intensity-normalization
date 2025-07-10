"""Image adapter implementations for universal image interface."""

import os
from typing import Any, cast

import nibabel as nib
import nibabel.spatialimages
import numpy as np
import numpy.typing as npt

from intensity_normalization.domain.exceptions import ImageLoadError
from intensity_normalization.domain.protocols import ImageProtocol


class NumpyImageAdapter:
    """Adapter for raw numpy arrays as images."""

    def __init__(self, data: npt.NDArray[np.floating], metadata: dict[str, Any] | None = None) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._metadata = metadata or {}

    def get_data(self) -> npt.NDArray[np.floating]:
        return self._data.copy()

    def with_data(self, data: npt.NDArray[np.floating]) -> "NumpyImageAdapter":
        return NumpyImageAdapter(data, self._metadata)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def save(self, path: str | os.PathLike) -> None:
        """Save as numpy .npy file."""
        np.save(path, self._data)


class NibabelImageAdapter:
    """Adapter for nibabel medical images."""

    def __init__(self, nib_image: nibabel.spatialimages.SpatialImage) -> None:
        self._nib_image = nib_image

    def get_data(self) -> npt.NDArray[np.floating]:
        return cast("npt.NDArray[np.floating]", self._nib_image.get_fdata())

    def with_data(self, data: npt.NDArray[np.floating]) -> "NibabelImageAdapter":
        # Create new nibabel image with same affine and header
        new_img = nib.nifti1.Nifti1Image(data, self._nib_image.affine, self._nib_image.header)
        return NibabelImageAdapter(new_img)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._nib_image.shape

    def save(self, path: str | os.PathLike) -> None:
        """Save in original nibabel format."""
        nib.loadsave.save(self._nib_image, path)

    @property
    def affine(self) -> npt.NDArray[np.floating] | None:
        return self._nib_image.affine

    @property
    def header(self):
        return self._nib_image.header


def create_image(
    source: str | os.PathLike | npt.NDArray | nibabel.spatialimages.SpatialImage,
) -> ImageProtocol:
    """Factory function to create appropriate image adapter."""
    if isinstance(source, str | os.PathLike):
        # Load from file - nibabel handles format detection
        try:
            nib_img = cast("nibabel.spatialimages.SpatialImage", nib.loadsave.load(source))
            return NibabelImageAdapter(nib_img)
        except Exception as e:
            raise ImageLoadError(f"Failed to load image from {source}: {e}") from e

    elif isinstance(source, np.ndarray):
        return NumpyImageAdapter(source)

    elif hasattr(source, "get_fdata"):  # nibabel image
        return NibabelImageAdapter(source)

    else:
        raise ImageLoadError(f"Unsupported image source type: {type(source)}")


# Supported nibabel formats (for CLI validation)
SUPPORTED_EXTENSIONS = {
    ".nii",
    ".nii.gz",  # NIfTI
    ".hdr",
    ".img",  # ANALYZE
    ".mgz",
    ".mgh",  # FreeSurfer
    ".mnc",  # MINC
    ".par",
    ".rec",  # Philips PAR/REC
}
