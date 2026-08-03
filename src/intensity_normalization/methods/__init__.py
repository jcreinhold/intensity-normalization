"""Normalization methods.

Two kinds of methods, reflected in the shape of the API:

- **Individual methods** (:func:`zscore`, :func:`fcm`, :func:`kde`,
  :func:`whitestripe`) are plain functions of one image: their parameters are
  estimated from the image being normalized, so there is nothing to fit,
  persist, or reuse.
- **Population methods** (:mod:`nyul`, :mod:`lsq`, :mod:`ravel`) learn a
  transform from a set of images: ``fit(images)`` returns a fitted transform
  object that is callable and savable, and can be applied to new images.
"""

from intensity_normalization.methods.fcm import fcm
from intensity_normalization.methods.kde import kde
from intensity_normalization.methods.whitestripe import whitestripe
from intensity_normalization.methods.zscore import zscore

__all__ = ["fcm", "kde", "whitestripe", "zscore"]
