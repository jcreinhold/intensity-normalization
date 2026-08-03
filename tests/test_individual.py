"""Correctness tests for individual normalizers on a phantom with known statistics."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from intensity_normalization import fcm, kde, whitestripe, zscore
from intensity_normalization.errors import IntensityNormalizationError


def test_zscore_standardizes_foreground(phantom) -> None:
    image, mask, _ = phantom
    out = zscore(image, mask)
    assert out[mask].mean() == pytest.approx(0.0, abs=1e-5)
    assert out[mask].std() == pytest.approx(1.0, abs=1e-5)


def test_zscore_scales_to_norm_value(phantom) -> None:
    image, mask, _ = phantom
    out = zscore(image, mask, norm_value=10.0)
    assert out[mask].std() == pytest.approx(10.0, abs=1e-4)


def test_fcm_wm_mean_to_norm_value(phantom) -> None:
    image, mask, labels = phantom
    out = fcm(image, mask, tissue="wm")
    assert out[labels == 3].mean() == pytest.approx(1.0, abs=0.15)


def test_fcm_deterministic(phantom) -> None:
    image, mask, _ = phantom
    assert np.array_equal(fcm(image, mask), fcm(image, mask))


def test_fcm_bad_tissue(phantom) -> None:
    image, mask, _ = phantom
    with pytest.raises(ValueError, match="Unknown tissue"):
        fcm(image, mask, tissue="bogus")


def test_kde_tissue_mode_to_norm_value(phantom) -> None:
    image, mask, _ = phantom
    out = kde(image, mask, norm_value=2.0)
    # the wm peak (largest tissue mean, last mode on t1) should sit at 2.0
    from intensity_normalization import histogram

    mode = histogram.tissue_mode(out[mask], modality="t1")
    assert mode == pytest.approx(2.0, abs=0.1)


def test_kde_peak_escape_hatch(phantom) -> None:
    image, mask, _ = phantom
    out = kde(image, mask, modality="other", peak="first", norm_value=5.0)
    from intensity_normalization import histogram

    mode = histogram.first_mode(out[mask])
    assert mode == pytest.approx(5.0, abs=0.5)


def test_kde_bad_modality(phantom) -> None:
    image, mask, _ = phantom
    with pytest.raises(ValueError, match="Unknown modality"):
        kde(image, mask, modality="bogus")


def test_whitestripe_stripe_statistics(phantom) -> None:
    image, mask, _ = phantom
    out = whitestripe(image, mask)
    # the white stripe centers on the wm mode: after normalization it is at 0
    from intensity_normalization import histogram

    mode = histogram.last_mode(out[mask])
    assert mode == pytest.approx(0.0, abs=0.2)


def test_type_preservation_numpy(phantom) -> None:
    image, mask, _ = phantom
    out = whitestripe(image, mask)
    assert isinstance(out, np.ndarray)
    assert out.shape == image.shape


def test_type_preservation_nibabel(phantom) -> None:
    image, mask, _ = phantom
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nii = nib.Nifti1Image(image, affine)
    nii_mask = nib.Nifti1Image(mask.astype(np.float32), affine)
    out = zscore(nii, nii_mask)
    assert isinstance(out, nib.Nifti1Image)
    assert np.allclose(out.affine, affine)


def test_no_mask_estimates_foreground(phantom) -> None:
    image, _, _ = phantom  # background is exactly 0
    out = zscore(image)
    fg = image > 0
    assert out[fg].std() == pytest.approx(1.0, abs=1e-5)


def test_empty_mask_actionable_error(phantom) -> None:
    image, _, _ = phantom
    with pytest.raises(IntensityNormalizationError, match="foreground is empty"):
        zscore(image, np.zeros_like(image))


def test_mask_shape_mismatch_error(phantom) -> None:
    image, _, _ = phantom
    with pytest.raises(IntensityNormalizationError, match="does not match"):
        zscore(image, np.ones((4, 4, 4), np.float32))


def test_unsupported_image_type() -> None:
    with pytest.raises(TypeError, match="Unsupported image type"):
        zscore("not-an-image")  # type: ignore[arg-type]
