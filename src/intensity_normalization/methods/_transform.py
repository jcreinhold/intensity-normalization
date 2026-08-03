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

from intensity_normalization._image import ImageLike

__all__ = ["FittedTransform"]


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

    def __call__(
        self,
        image: ImageLike,
        /,
        mask: ImageLike | None = None,
        **kwargs: typing.Any,
    ) -> ImageLike:
        return self.transform(image, mask, **kwargs)

    @abc.abstractmethod
    def transform(self, image: ImageLike, /, mask: ImageLike | None = None) -> ImageLike:
        """Apply the learned transform to one image."""

    @abc.abstractmethod
    def _state_dict(self) -> dict[str, np.ndarray]:
        """Learned parameters as numpy arrays for serialization."""

    @classmethod
    @abc.abstractmethod
    def _from_state_dict(cls, state: dict[str, np.ndarray]) -> FittedTransform:
        """Rebuild a transform from its state dict."""

    def save(self, path: str | PathLike[str], /) -> None:
        """Save the fitted transform to ``path`` (``.npz``)."""
        state: dict[str, np.ndarray] = {
            "_method": np.array(self.method),
            "_format_version": np.array(self.format_version),
            **self._state_dict(),
        }
        np.savez(path, **state)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | PathLike[str], /) -> FittedTransform:
        """Load a transform saved with :meth:`save`.

        Raises:
            ValueError: the file was saved by a different method or format
                version than this class.
        """
        with np.load(path) as data:
            state = {k: data[k] for k in data.files}
        method = str(state.pop("_method"))
        version = int(state.pop("_format_version"))
        if method != cls.method:
            raise ValueError(
                f"{path} holds a {method!r} transform, not {cls.method!r}. "
                f"Load it with the {method} transform class instead."
            )
        if version != cls.format_version:
            raise ValueError(
                f"{path} uses {method} format version {version}; "
                f"this version of intensity-normalization reads version {cls.format_version}."
            )
        return cls._from_state_dict(state)
