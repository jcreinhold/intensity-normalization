"""Correctness tests for the in-house fuzzy c-means."""

from __future__ import annotations

import numpy as np
import pytest

from intensity_normalization.methods._fcm import fuzzy_cmeans, predict_memberships


def test_recovers_known_centers() -> None:
    rng = np.random.default_rng(42)
    data = np.concatenate([rng.normal(100, 15, 4000), rng.normal(300, 15, 6000), rng.normal(500, 15, 8000)]).astype(
        np.float32
    )
    centers, memberships = fuzzy_cmeans(data, 3, seed=0)
    assert centers == pytest.approx([100, 300, 500], abs=5.0)
    # centers sorted ascending; memberships sum to 1 per sample
    assert np.all(np.diff(centers) > 0)
    assert np.allclose(memberships.sum(axis=0), 1.0)
    # hard assignment matches ground truth
    hard = memberships.argmax(axis=0)
    truth = np.repeat([0, 1, 2], [4000, 6000, 8000])
    assert (hard == truth).mean() > 0.98


def test_deterministic_with_seed() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 1000).astype(np.float32)
    c1, u1 = fuzzy_cmeans(data, 2, seed=7)
    c2, u2 = fuzzy_cmeans(data, 2, seed=7)
    assert np.array_equal(c1, c2)
    assert np.array_equal(u1, u2)


def test_predict_memberships_exact_center() -> None:
    centers = np.array([100.0, 300.0, 500.0], dtype=np.float32)
    memberships = predict_memberships(np.array([100.0]), centers)
    assert memberships[0, 0] == pytest.approx(1.0)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        fuzzy_cmeans(np.array([1.0, 2.0]), 0)
    with pytest.raises(ValueError, match="m must be"):
        fuzzy_cmeans(np.array([1.0, 2.0, 3.0]), 2, m=1.0)
    with pytest.raises(ValueError, match="distinct values"):
        fuzzy_cmeans(np.ones(10, dtype=np.float32), 3)
