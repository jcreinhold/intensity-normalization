"""Metadata and information-preservation tests.

The contract: a normalized image carries the *same spatial metadata* as its
input — identical affine, preserved qform/sform codes, unmutated source header
— and its stored datatype reflects the normalized (float32) data, so a
save/reload round-trip never loses information.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from phantoms import make_phantom

import intensity_normalization as inorm
from intensity_normalization import io
from intensity_normalization.errors import IntensityNormalizationError


@st.composite
def rigid_affine(draw):
    """A realistic scanner affine: drawn spacing, axis flip, and translation."""
    sx, sy, sz = draw(st.tuples(*[st.floats(0.5, 4.0)] * 3))
    tx, ty, tz = draw(st.tuples(*[st.floats(-150.0, 150.0)] * 3))
    flip = draw(st.integers(0, 1)) * -2 + 1
    affine = np.diag([flip * sx, sy, sz, 1.0])
    affine[:3, 3] = [tx, ty, tz]
    return affine


def _nifti_pair(dtype, affine):
    import nibabel as nib

    image, mask, _ = make_phantom((20, 20, 20))
    scaled = image * (1000.0 / image.max())  # fill integer dtypes meaningfully
    data = scaled.astype(dtype)
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.dtype(dtype))
    hdr.set_qform(affine, code=1)
    hdr.set_sform(affine, code=1)
    img = nib.Nifti1Image(data, affine, hdr)
    msk = nib.Nifti1Image((mask).astype(dtype), affine, hdr.copy())
    return img, msk


DTYPES = st.sampled_from([np.int16, np.int32, np.float32, np.float64])


@given(dtype=DTYPES, affine=rigid_affine())
@settings(max_examples=8, deadline=None)
def test_affine_and_header_survive_normalization(dtype, affine) -> None:
    import nibabel as nib

    img, msk = _nifti_pair(dtype, affine)
    out = inorm.zscore(img, msk)
    assert isinstance(out, nib.Nifti1Image)
    # affine identical, codes preserved
    assert np.array_equal(out.affine, img.affine)
    assert out.header["qform_code"] == img.header["qform_code"]
    assert out.header["sform_code"] == img.header["sform_code"]
    # stored dtype follows the normalized data, not the source dtype
    assert out.header.get_data_dtype() == np.dtype(np.float32)
    # the source image's header is never mutated
    assert img.header.get_data_dtype() == np.dtype(dtype)


@given(dtype=DTYPES, affine=rigid_affine())
@settings(max_examples=8, deadline=None)
def test_save_reload_roundtrip_preserves_data(dtype, affine) -> None:
    img, msk = _nifti_pair(dtype, affine)
    out = inorm.zscore(img, msk)
    with tempfile.TemporaryDirectory() as td:
        path = io.save_image(out, pathlib.Path(td) / "out.nii.gz")
        reloaded = io.load_image(path)
        # read through the proxy before the tempdir goes away
        assert np.allclose(np.asanyarray(reloaded.dataobj), np.asanyarray(out.dataobj))
        assert np.allclose(reloaded.affine, affine)


def test_scaled_source_image_no_residual_slope(tmp_path) -> None:
    """A source with scl_slope: values are read scaled; output stores raw floats."""
    import nibabel as nib

    image, mask, _ = make_phantom((16, 16, 16))
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.int16)
    hdr["scl_slope"] = 2.0
    hdr["scl_inter"] = 10.0
    img = nib.Nifti1Image(image.astype(np.int16), np.eye(4), hdr)
    msk = nib.Nifti1Image(mask.astype(np.int16), np.eye(4))
    out = inorm.zscore(img, msk)
    path = io.save_image(out, tmp_path / "o.nii.gz")
    reloaded = io.load_image(path)
    assert np.allclose(np.asanyarray(reloaded.dataobj), np.asanyarray(out.dataobj))


def test_4d_output_keeps_affine(tmp_path) -> None:
    import nibabel as nib

    image, mask, _ = make_phantom((16, 16, 16))
    affine = np.diag([1.5, 2.0, 3.0, 1.0])
    img = nib.Nifti1Image(image, affine)
    msk = nib.Nifti1Image(mask.astype(np.float32), affine)
    out = inorm.tissue_membership(img, msk)
    assert out.shape == (*image.shape, 3)
    assert np.array_equal(out.affine, affine)


def test_mask_affine_mismatch_rejected() -> None:
    import nibabel as nib

    image, mask, _ = make_phantom((16, 16, 16))
    img = nib.Nifti1Image(image, np.diag([2.0, 2.0, 2.0, 1.0]))
    msk = nib.Nifti1Image(mask.astype(np.float32), np.eye(4))  # same shape, other space
    with pytest.raises(IntensityNormalizationError, match="different spaces"):
        inorm.zscore(img, msk)


def test_numpy_mask_with_nibabel_image_allowed() -> None:
    import nibabel as nib

    image, mask, _ = make_phantom((16, 16, 16))
    img = nib.Nifti1Image(image, np.eye(4))
    out = inorm.zscore(img, mask.astype(np.float32))
    assert isinstance(out, nib.Nifti1Image)
