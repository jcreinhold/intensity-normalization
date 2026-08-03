"""Regression tests: the smallest reproducer for each bug fixed in v4 development.

Each test names the failure it guards against; without the corresponding fix,
it fails.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import make_phantom, make_population

import intensity_normalization as inorm
from intensity_normalization import io


def test_regression_int16_source_truncates_on_save(tmp_path) -> None:
    """int16 source header used to truncate float32 normalized data on save."""
    import nibabel as nib

    image, mask, _ = make_phantom((16, 16, 16))
    img = nib.Nifti1Image(image.astype(np.int16), np.eye(4))
    out = inorm.zscore(img, nib.Nifti1Image(mask.astype(np.int16), np.eye(4)))
    path = io.save_image(out, tmp_path / "o.nii.gz")
    reloaded = np.asanyarray(io.load_image(path).dataobj)
    assert reloaded.dtype != np.int16
    assert np.allclose(reloaded, np.asanyarray(out.dataobj))


def test_regression_constant_foreground_no_nan() -> None:
    """Degenerate (constant) foregrounds must raise clean errors, not emit NaN."""
    image = np.full((8, 8, 8), 100.0, dtype=np.float32)
    mask = image > 0
    for fn in (inorm.zscore, inorm.fcm, inorm.whitestripe):
        with pytest.raises((inorm.IntensityNormalizationError, ValueError)):
            fn(image, mask)


def test_regression_nyul_degenerate_percentile_grid_rejected() -> None:
    """A non-increasing landmark grid used to reach interp1d and produce NaN."""
    images, masks = make_population(3)
    with pytest.raises(ValueError, match="strictly increasing"):
        inorm.nyul.fit(images, masks, landmarks=[50.0, 10.0, 90.0])


def test_regression_discrete_image_memberships_finite() -> None:
    """Voxels exactly at cluster centers (discrete images) caused NaN memberships."""
    image = np.zeros((12, 12, 12), np.float32)
    image[2:-2, 2:-2, 2:-2] = 100.0
    image[3:-3, 3:-3, 3:-3] = 300.0
    image[4:-4, 4:-4, 4:-4] = 500.0
    tm = inorm.tissue_membership(image, image > 0)
    assert np.isfinite(tm).all()
    fg = image > 0
    assert np.allclose(tm[fg].sum(axis=1), 1.0, atol=1e-4)
