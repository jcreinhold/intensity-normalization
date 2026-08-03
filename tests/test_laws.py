"""Law-shaped property tests over the public API.

Each test states a contract the package must honor for *any* valid input, then
checks it over generated cases. The oracle is the law, not an example.

Laws covered:
- type/shape preservation (numpy in -> numpy out, same shape, float32)
- finiteness: valid inputs never produce NaN/Inf
- scale equivariance: normalizing c*x equals normalizing x (c > 0)
- shift invariance (zscore)
- determinism: same seed -> bit-identical output
- monotonicity of the learned nyul mapping
- serialization round-trip: save -> load -> apply == apply
- rejection: invalid arguments and forbidden states fail cleanly
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import intensity_normalization as inorm
from intensity_normalization.errors import IntensityNormalizationError

# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

TWO_POP = st.tuples(
    st.floats(50.0, 500.0),
    st.floats(40.0, 200.0),
    st.floats(40.0, 200.0),
).map(lambda t: (t[0], t[0] + t[1], t[0] + t[1] + t[2]))  # sorted, well-separated means


@st.composite
def phantom_3d(
    draw,
    shape=st.tuples(st.integers(10, 14), st.integers(10, 14), st.integers(10, 14)),
    means=TWO_POP,
    noise=st.floats(5.0, 25.0),
    seed=st.integers(0, 10_000),
):
    """A T1-w-like phantom: background 0, three tissue classes at drawn means."""
    shape_ = draw(shape)
    m1, m2, m3 = draw(means)
    sigma = draw(noise)
    rng = np.random.default_rng(draw(seed))
    labels = np.zeros(shape_, np.uint8)
    a = max(2, shape_[0] // 4)
    labels[a:-a, a:-a, a:-a] = 1
    b = max(1, shape_[0] // 8)
    ring = labels[b:-b, b:-b, b:-b]
    ring[ring == 0] = 2
    labels[labels == 1] = 3  # inner core is the smallest class (wm-like)
    image = np.zeros(shape_, np.float32)
    for k, mean in ((1, m1), (2, m2), (3, m3)):
        image[labels == k] = rng.normal(mean, sigma, int((labels == k).sum()))
    return image, labels > 0


@st.composite
def population_3d(draw):
    """The same phantom at drawn global scales (a cross-scanner population)."""
    image, mask = draw(phantom_3d())
    scales = draw(st.lists(st.floats(0.5, 2.0), min_size=3, max_size=5, unique=True))
    images = [image * s for s in scales]
    masks = [mask.astype(np.float32)] * len(images)
    return images, masks


POSITIVE_SCALE = st.floats(0.25, 4.0)

FAST = settings(max_examples=15, deadline=None)
SLOW = settings(max_examples=6, deadline=None)

# ---------------------------------------------------------------------------
# type/shape/finiteness laws
# ---------------------------------------------------------------------------


@given(pair=phantom_3d())
@FAST
def test_zscore_type_shape_finite(pair) -> None:
    image, mask = pair
    out = inorm.zscore(image, mask)
    assert isinstance(out, np.ndarray) and out.shape == image.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


@given(pair=phantom_3d())
@SLOW
def test_all_individual_methods_finite(pair) -> None:
    image, mask = pair
    for fn in (inorm.kde, inorm.whitestripe, inorm.fcm):
        out = fn(image, mask)
        assert np.isfinite(out).all()
        assert out.shape == image.shape


# ---------------------------------------------------------------------------
# equivariance laws
# ---------------------------------------------------------------------------


@given(pair=phantom_3d(), scale=POSITIVE_SCALE)
@FAST
def test_zscore_affine_invariant(pair, scale) -> None:
    """zscore(a*x + b) == zscore(x): it removes exactly these degrees of freedom."""
    image, mask = pair
    base = inorm.zscore(image, mask)
    shifted = inorm.zscore(image * scale + 17.0, mask)
    assert np.allclose(base, shifted, atol=1e-3)


@given(pair=phantom_3d(), scale=POSITIVE_SCALE)
@SLOW
def test_kde_scale_equivariant(pair, scale) -> None:
    image, mask = pair
    base = inorm.kde(image, mask)
    scaled = inorm.kde(image * scale, mask)
    # atol not rtol: normalized values cross zero, so relative error is meaningless
    assert np.allclose(base, scaled, atol=0.05)


@given(pair=phantom_3d(), scale=POSITIVE_SCALE)
@SLOW
def test_whitestripe_scale_equivariant(pair, scale) -> None:
    image, mask = pair
    base = inorm.whitestripe(image, mask)
    scaled = inorm.whitestripe(image * scale, mask)
    assert np.allclose(base, scaled, atol=0.05)


@given(pair=phantom_3d(), scale=POSITIVE_SCALE)
@SLOW
def test_fcm_scale_equivariant(pair, scale) -> None:
    image, mask = pair
    base = inorm.fcm(image, mask)
    scaled = inorm.fcm(image * scale, mask)
    assert np.allclose(base, scaled, atol=0.05)


# ---------------------------------------------------------------------------
# determinism laws
# ---------------------------------------------------------------------------


@given(pair=phantom_3d())
@SLOW
def test_methods_deterministic_with_default_seed(pair) -> None:
    image, mask = pair
    for fn in (inorm.kde, inorm.whitestripe, inorm.fcm):
        assert np.array_equal(fn(image, mask), fn(image, mask))
    assert np.array_equal(inorm.tissue_membership(image, mask), inorm.tissue_membership(image, mask))


# ---------------------------------------------------------------------------
# population-method laws
# ---------------------------------------------------------------------------


@given(pop=population_3d())
@SLOW
def test_fitted_transforms_immutable(pop) -> None:
    """A fitted transform cannot be corrupted: frozen fields, read-only arrays.

    This guarantees a saved transform always matches the in-memory one.
    """
    import dataclasses

    images, masks = pop
    tx_nyul = inorm.nyul.fit(images, masks)
    tx_lsq = inorm.lsq.fit(images, masks)
    for tx in (tx_nyul, tx_lsq):
        with pytest.raises(dataclasses.FrozenInstanceError):
            tx.norm_value = 2.0  # type: ignore[attr-defined, misc]
    with pytest.raises(ValueError, match="read-only"):
        tx_nyul.standard_scale[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        tx_lsq.standard_tissue_means[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        tx_lsq.reference_membership[..., 0] = 0.0


@given(pop=population_3d())
@SLOW
def test_nyul_mapping_monotonic(pop) -> None:
    """The learned piecewise-linear map never inverts intensity order."""
    images, masks = pop
    tx = inorm.nyul.fit(images, masks)
    probes = np.linspace(1.0, 1000.0, 200, dtype=np.float32)
    mapped = tx(probes.reshape(5, 8, 5)).ravel()
    assert np.all(np.diff(mapped) >= -1e-4)


@given(pop=population_3d())
@SLOW
def test_nyul_roundtrip_law(pop) -> None:
    images, masks = pop
    tx = inorm.nyul.fit(images, masks)
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "tx.npz"
        tx.save(path)
        reloaded = inorm.NyulTransform.load(path)
    assert np.array_equal(tx(images[0], masks[0]), reloaded(images[0], masks[0]))


@given(pop=population_3d())
@SLOW
def test_lsq_roundtrip_law(pop) -> None:
    images, masks = pop
    tx = inorm.lsq.fit(images, masks)
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "tx.npz"
        tx.save(path)
        reloaded = inorm.LSQTransform.load(path)
    assert np.array_equal(tx(images[0], masks[0]), reloaded(images[0], masks[0]))


# ---------------------------------------------------------------------------
# rejection laws
# ---------------------------------------------------------------------------


@given(bad=st.text(min_size=1, max_size=8).filter(lambda s: s not in {"t1", "t2", "flair", "pd", "md", "other"}))
@FAST
def test_unknown_modality_rejected(bad) -> None:
    image = np.ones((6, 6, 6), np.float32) * 100
    with pytest.raises(ValueError, match="Unknown modality"):
        inorm.kde(image, modality=bad)


@given(value=st.floats(1.0, 1000.0))
@FAST
def test_constant_foreground_rejected_cleanly(value) -> None:
    """Constant foreground: every method must fail with a clean error, never NaN."""
    image = np.zeros((8, 8, 8), np.float32)
    image[2:-2, 2:-2, 2:-2] = value
    mask = image > 0
    for fn in (inorm.zscore, inorm.fcm, inorm.whitestripe):
        with pytest.raises((IntensityNormalizationError, ValueError)):
            fn(image, mask)
