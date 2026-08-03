"""Base class for fitted population transforms (private).

A population method's ``fit()`` returns one of these: a fully-fitted, callable,
serializable transform. Because construction *is* fitting, no unfitted state
can exist and there is nothing like a ``NotFittedError`` to handle.
"""

from __future__ import annotations

import abc
import typing
from os import PathLike

import numpy as np

from intensity_normalization import _image
from intensity_normalization._image import BinaryMask, Image, IntensityArray, Mask

__all__ = ["FittedTransform", "_load_stamped", "_save_stamped"]


def _save_stamped(
    path: str | PathLike[str],
    method: str,
    version: int,
    state: dict[str, np.ndarray],
) -> None:
    """Write a state dict to a stamped ``.npz`` (single owner of the file format)."""
    np.savez_compressed(
        path,
        _method=np.array(method),
        _format_version=np.array(version),
        **state,  # ty: ignore[invalid-argument-type]  # **dict[str, NDArray] is valid here
    )


def _load_stamped(
    path: str | PathLike[str],
    method: str,
    version: int,
) -> dict[str, np.ndarray]:
    """Read a stamped ``.npz``, verifying the method stamp and format version."""
    with np.load(path) as data:
        state = {k: data[k] for k in data.files}
    found_method = str(state.pop("_method"))
    found_version = int(state.pop("_format_version"))
    if found_method != method:
        raise ValueError(
            f"{path} holds {found_method!r} artifacts, not {method!r}. Load it with the matching class instead."
        )
    if found_version != version:
        raise ValueError(
            f"{path} uses {method} format version {found_version}; "
            f"this version of intensity-normalization reads version {version}."
        )
    return state


class FittedTransform(abc.ABC):
    """A normalization transform learned from a population of images.

    Callable and type-preserving: pass a numpy array or nibabel image, get the
    same type back. Serializable to a stamped ``.npz`` via :meth:`save` /
    :meth:`load`.
    """

    #: method identifier stamped into saved files
    method: typing.ClassVar[str]
    #: serialization format version, bumped on incompatible changes
    format_version: typing.ClassVar[int] = 1

    def __call__(self, image: Image, mask: Mask | None = None) -> Image:
        """Sugar for :meth:`transform`."""
        return self.transform(image, mask)

    def transform(self, image: Image, mask: Mask | None = None) -> Image:
        """Apply the learned transform to one image (numpy array or nibabel image)."""
        data, meta = _image.unwrap(image)
        foreground = _image.resolve_foreground(data, _image.unwrap_mask(image, mask))
        out = self.transform_array(data, foreground)
        return _image.restore(meta, out)

    @abc.abstractmethod
    def transform_array(self, data: IntensityArray, foreground: BinaryMask) -> IntensityArray:
        """Apply the learned transform to an intensity array within ``foreground``."""

    @abc.abstractmethod
    def _state_dict(self) -> dict[str, np.ndarray]:
        """Learned parameters as numpy arrays for serialization."""

    @classmethod
    @abc.abstractmethod
    def _from_state_dict(cls, state: dict[str, np.ndarray]) -> FittedTransform:
        """Rebuild a transform from its state dict."""

    def save(self, path: str | PathLike[str]) -> None:
        """Save the fitted transform to ``path`` (``.npz``)."""
        _save_stamped(path, self.method, self.format_version, self._state_dict())

    @classmethod
    def load(cls, path: str | PathLike[str]) -> FittedTransform:
        """Load a transform saved with :meth:`save`.

        Raises:
            ValueError: the file was saved by a different method or format
                version than this class.
        """
        return cls._from_state_dict(_load_stamped(path, cls.method, cls.format_version))
