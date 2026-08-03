"""intensity-normalization: normalize the intensities of MR images.

Individual methods are plain functions::

    import intensity_normalization as inorm
    normed = inorm.whitestripe(image, mask=mask)

Population methods learn a savable transform from a set of images::

    tx = inorm.nyul.fit(training_images, masks=masks)
    normed = tx(new_image)
    tx.save("nyul.npz")

numpy arrays in -> numpy out; nibabel images in -> nibabel images out
(affine/header preserved).
"""

import importlib.metadata
import logging

from intensity_normalization import histogram, io
from intensity_normalization._image import (
    BinaryMask,
    ForegroundIntensities,
    Image,
    IntensityArray,
    Mask,
    MaskArray,
)
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods import fcm, kde, lsq, nyul, ravel, whitestripe, zscore
from intensity_normalization.methods.lsq import LSQTransform
from intensity_normalization.methods.nyul import NyulTransform
from intensity_normalization.methods.ravel import RavelResult
from intensity_normalization.tools.ants import coregister, preprocess
from intensity_normalization.tools.plot import plot_histograms
from intensity_normalization.tools.tissue import tissue_membership

__version__ = importlib.metadata.version("intensity-normalization")

__all__ = [
    "BinaryMask",
    "ForegroundIntensities",
    "Image",
    "IntensityArray",
    "IntensityNormalizationError",
    "LSQTransform",
    "Mask",
    "MaskArray",
    "NyulTransform",
    "RavelResult",
    "__version__",
    "coregister",
    "fcm",
    "histogram",
    "io",
    "kde",
    "lsq",
    "nyul",
    "plot_histograms",
    "preprocess",
    "ravel",
    "tissue_membership",
    "whitestripe",
    "zscore",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
