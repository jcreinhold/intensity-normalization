"""Tests for tools (tissue membership, plotting, io) — ants tools covered separately."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from intensity_normalization import io, tissue_membership
from intensity_normalization.errors import IntensityNormalizationError


def test_tissue_membership_shape_and_order(phantom) -> None:
    image, mask, labels = phantom
    tm = tissue_membership(image, mask)
    assert tm.shape == (*image.shape, 3)
    fg = mask
    assert np.allclose(tm[fg].sum(axis=1), 1.0, atol=1e-4)
    # csf class should have highest membership where labels == 1
    assert tm[..., 0][labels == 1].mean() > 0.5


def test_tissue_membership_hard(phantom) -> None:
    image, mask, _ = phantom
    hard = tissue_membership(image, mask, hard_segmentation=True)
    assert set(np.unique(hard)) == {0.0, 1.0, 2.0, 3.0}


def test_tissue_membership_nibabel(phantom) -> None:
    image, mask, _ = phantom
    nii = nib.Nifti1Image(image, np.eye(4))
    out = tissue_membership(nii, nib.Nifti1Image(mask.astype(np.float32), np.eye(4)))
    assert isinstance(out, nib.Nifti1Image)
    assert out.shape == (*image.shape, 3)


def test_plot_histograms(phantom, tmp_path, monkeypatch) -> None:
    import matplotlib

    monkeypatch.setattr(matplotlib.pyplot, "show", lambda: None)
    from intensity_normalization import plot_histograms, zscore

    image, mask, _ = phantom
    out = tmp_path / "hist.png"
    fig = plot_histograms([image, zscore(image, mask)], [mask, mask], output=out)
    assert out.exists()
    assert fig is not None


def test_io_roundtrip(tmp_path) -> None:
    image = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    path = io.save_image(nib.Nifti1Image(image, np.eye(4)), tmp_path / "x.nii.gz")
    loaded = io.load_image(path)
    assert np.allclose(np.asarray(loaded.dataobj), image)


def test_io_split_filename() -> None:
    _, base, ext = io.split_filename("a/b/img.nii.gz")
    assert base == "img" and ext == ".nii.gz"
    _, base2, ext2 = io.split_filename("img.mgz")
    assert base2 == "img" and ext2 == ".mgz"


def test_io_find_images_and_match_masks(nifti_dir) -> None:
    image_dir, mask_dir = nifti_dir
    paths = io.find_images(image_dir)
    assert len(paths) == 3
    masks = io.match_masks(paths, mask_dir)
    assert all(m.exists() for m in masks)
    empty = mask_dir.parent / "empty"
    empty.mkdir()
    with pytest.raises(IntensityNormalizationError, match="No neuroimages"):
        io.find_images(empty)
    with pytest.raises(IntensityNormalizationError, match="Not a directory"):
        io.find_images(mask_dir.parent / "nope")


def test_io_match_masks_missing(nifti_dir) -> None:
    image_dir, _ = nifti_dir
    with pytest.raises(IntensityNormalizationError, match="No mask named"):
        io.match_masks(io.find_images(image_dir), image_dir.parent)


def test_io_output_path() -> None:
    out = io.output_path("/data/sub0.nii.gz", suffix="ws", output_dir="/out")
    assert str(out) == "/out/sub0_ws.nii.gz"
