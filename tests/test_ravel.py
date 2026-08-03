"""Tests for RAVEL (registration-free path; ants paths are covered by marked tests)."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import make_population

from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods import ravel


def test_ravel_removes_across_image_trend() -> None:
    images, masks = make_population(5)
    result, normed = ravel.fit_transform(images, masks, register=False, membership_threshold=0.9)
    control = result.control_mask
    before = np.stack([_whitestriped(i, m)[control] for i, m in zip(images, masks, strict=True)]).std(axis=0)
    after = np.stack([n[control] for n in normed]).std(axis=0)
    assert after.mean() < before.mean() * 0.5


def _whitestriped(image, mask):
    from intensity_normalization import whitestripe

    return whitestripe(image, mask)


def test_ravel_shapes_preserved() -> None:
    images, masks = make_population(3)
    _, normed = ravel.fit_transform(images, masks, register=False, membership_threshold=0.9)
    assert all(n.shape == images[0].shape for n in normed)


def test_ravel_shape_mismatch_error() -> None:
    images, masks = make_population(2)
    bad = images[1][:-1]
    with pytest.raises(IntensityNormalizationError, match="same shape"):
        ravel.fit_transform([images[0], bad], [masks[0], masks[1][:-1]], register=False)


def test_ravel_save_load_roundtrip(tmp_path) -> None:
    images, masks = make_population(3)
    result, _ = ravel.fit_transform(images, masks, register=False, membership_threshold=0.9)
    path = tmp_path / "ravel.npz"
    result.save(path)
    loaded = ravel.RavelResult.load(path)
    assert np.array_equal(result.unwanted_factors, loaded.unwanted_factors)
    assert np.array_equal(result.control_mask, loaded.control_mask)


def test_ravel_register_and_csf_masks_conflict() -> None:
    images, masks = make_population(2)
    with pytest.raises(ValueError, match="masks_are_csf"):
        ravel.fit_transform(images, masks, register=True, masks_are_csf=True)


def test_ravel_float32_memory() -> None:
    images, masks = make_population(3)
    result, normed = ravel.fit_transform(images, masks, register=False, membership_threshold=0.9)
    assert result.control_voxels.dtype == np.float32
    assert all(n.dtype == np.float32 for n in normed)
