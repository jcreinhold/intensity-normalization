"""Tests for population methods: fit/transform semantics, streaming, serialization."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import make_population

from intensity_normalization import histogram
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods import lsq, nyul


def test_nyul_maps_landmarks_to_output_range(population) -> None:
    images, masks = population
    tx = nyul.fit(images, masks)
    out = tx(images[1], masks[1])
    fg = out[masks[1] > 0]
    assert np.percentile(fg, 1) == pytest.approx(1.0, abs=0.1)
    assert np.percentile(fg, 99) == pytest.approx(100.0, abs=0.1)


def test_nyul_aligns_population_histograms(population) -> None:
    images, masks = population
    _, normed = nyul.fit_transform(images, masks)
    modes = [histogram.tissue_mode(n[m > 0], modality="t1") for n, m in zip(normed, masks, strict=True)]
    assert np.std(modes) < 2.0  # raw modes range over ~2x; aligned should be tight


def test_nyul_save_load_roundtrip(population, tmp_path) -> None:
    images, masks = population
    tx = nyul.fit(images, masks)
    path = tmp_path / "nyul.npz"
    tx.save(path)
    tx2 = nyul.NyulTransform.load(path)
    assert np.array_equal(tx.standard_scale, tx2.standard_scale)
    assert np.array_equal(tx(images[0], masks[0]), tx2(images[0], masks[0]))


def test_nyul_load_wrong_method_error(population, tmp_path) -> None:
    images, masks = population
    tx = lsq.fit(images, masks)
    assert isinstance(tx, lsq.LSQTransform)
    path = tmp_path / "lsq.npz"
    tx.save(path)
    with pytest.raises(ValueError, match="not 'nyul'"):
        nyul.NyulTransform.load(path)


def test_nyul_empty_input() -> None:
    with pytest.raises(IntensityNormalizationError, match="No images"):
        nyul.fit([])


def test_nyul_mask_count_mismatch(population) -> None:
    images, masks = population
    with pytest.raises(ValueError, match="masks"):
        nyul.fit(images, masks[:2])


def test_lsq_aligns_tissue_means(population) -> None:
    images, masks = population
    tx = lsq.fit(images, masks)
    assert isinstance(tx, lsq.LSQTransform)
    gm_modes = []
    for image, mask in zip(images, masks, strict=True):
        normed = tx(image, mask)
        from intensity_normalization.methods.fcm import tissue_means

        means, _ = tissue_means(normed, mask > 0, seed=0)
        gm_modes.append(means[1])
    assert np.std(gm_modes) < 0.2  # tissue means pulled together vs ~2x spread raw


def test_lsq_save_load_roundtrip(population, tmp_path) -> None:
    images, masks = population
    tx = lsq.fit(images, masks)
    assert isinstance(tx, lsq.LSQTransform)
    path = tmp_path / "lsq.npz"
    tx.save(path)
    tx2 = lsq.LSQTransform.load(path)
    assert np.array_equal(tx.standard_tissue_means, tx2.standard_tissue_means)
    assert np.array_equal(tx(images[0], masks[0]), tx2(images[0], masks[0]))


def test_lsq_reference_membership(population, tmp_path) -> None:
    """The reference membership map is always fitted, stored, and persisted."""
    images, masks = population
    tx = lsq.fit(images, masks)
    tissue_map = tx.reference_membership
    assert tissue_map.shape == (*images[0].shape, 3)
    fg = masks[0] > 0
    assert np.allclose(tissue_map[fg].sum(axis=1), 1.0, atol=1e-4)
    # persisted: available after a save/load round-trip
    path = tmp_path / "tx.npz"
    tx.save(path)
    assert np.array_equal(tissue_map, lsq.LSQTransform.load(path).reference_membership)


def test_lsq_transform_with_membership(population) -> None:
    """Non-T1 path: transform with an externally supplied membership map."""
    images, masks = population
    tx = lsq.fit(images, masks)
    assert isinstance(tx, lsq.LSQTransform)
    from intensity_normalization import tissue_membership

    membership = tissue_membership(images[2], masks[2])
    out = tx(images[2] * 1.7, masks[2], membership=membership)
    assert out.shape == images[2].shape


def test_fit_deterministic(population) -> None:
    images, masks = population
    tx1 = lsq.fit(images, masks)
    tx2 = lsq.fit(images, masks)
    assert isinstance(tx1, lsq.LSQTransform) and isinstance(tx2, lsq.LSQTransform)
    assert np.array_equal(tx1.standard_tissue_means, tx2.standard_tissue_means)


def test_streaming_fit_memory(population) -> None:
    """nyul fit must not retain image data — only the standard scale."""
    images, masks = make_population(5)
    tx = nyul.fit(images, masks)
    assert tx.standard_scale.nbytes < 1024
