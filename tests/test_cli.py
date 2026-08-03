"""End-to-end CLI tests via typer's runner (no display, no ants)."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from intensity_normalization import io
from intensity_normalization.cli import app, main  # noqa: F401  (main registers commands)
from intensity_normalization.cli import normalize as cli_normalize
from intensity_normalization.cli import tools as cli_tools

cli_normalize.register(app)
cli_tools.register(app)

runner = CliRunner()


def test_zscore_single(nifti_dir, tmp_path) -> None:
    image_dir, mask_dir = nifti_dir
    out = tmp_path / "out.nii.gz"
    result = runner.invoke(
        app, ["zscore", str(image_dir / "sub0.nii.gz"), "-m", str(mask_dir / "sub0.nii.gz"), "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    data = np.asarray(io.load_image(out).dataobj)
    mask = np.asarray(io.load_image(mask_dir / "sub0.nii.gz").dataobj) > 0
    assert data[mask].std() == pytest.approx(1.0, abs=1e-4)


def test_whitestripe_batch(nifti_dir, tmp_path) -> None:
    image_dir, mask_dir = nifti_dir
    out_dir = tmp_path / "ws"
    result = runner.invoke(
        app,
        ["whitestripe", str(image_dir), "--mask-dir", str(mask_dir), "--output-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    assert len(list(out_dir.glob("*_whitestripe.nii.gz"))) == 3


def test_nyul_save_and_load_state(nifti_dir, tmp_path) -> None:
    image_dir, mask_dir = nifti_dir
    state = tmp_path / "nyul.npz"
    out1, out2 = tmp_path / "n1", tmp_path / "n2"
    result = runner.invoke(
        app, ["nyul", str(image_dir), "-m", str(mask_dir), "-o", str(out1), "--save-state", str(state)]
    )
    assert result.exit_code == 0, result.output
    assert state.exists()
    result = runner.invoke(
        app, ["nyul", str(image_dir), "-m", str(mask_dir), "-o", str(out2), "--load-state", str(state)]
    )
    assert result.exit_code == 0, result.output
    a = np.asarray(io.load_image(out1 / "sub0_nyul.nii.gz").dataobj)
    b = np.asarray(io.load_image(out2 / "sub0_nyul.nii.gz").dataobj)
    assert np.array_equal(a, b)


def test_lsq_with_tissue_maps(nifti_dir, tmp_path) -> None:
    image_dir, mask_dir = nifti_dir
    out_dir = tmp_path / "lsq"
    result = runner.invoke(
        app,
        ["lsq", str(image_dir), "-m", str(mask_dir), "-o", str(out_dir), "--save-tissue-maps"],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "sub0_tissue_membership.nii.gz").exists()


def test_tissue_membership_cmd(nifti_dir, tmp_path) -> None:
    image_dir, mask_dir = nifti_dir
    out = tmp_path / "tm.nii.gz"
    result = runner.invoke(
        app,
        ["tissue-membership", str(image_dir / "sub0.nii.gz"), "-m", str(mask_dir / "sub0.nii.gz"), "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert io.load_image(out).shape[-1] == 3


def test_plot_histograms_cmd(nifti_dir, tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    image_dir, mask_dir = nifti_dir
    out = tmp_path / "hist.png"
    result = runner.invoke(app, ["plot-histograms", str(image_dir), "-m", str(mask_dir), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_error_is_clean_not_traceback(nifti_dir, tmp_path) -> None:
    image_dir, _ = nifti_dir
    result = runner.invoke(app, ["zscore", str(image_dir / "sub0.nii.gz"), "-m", str(tmp_path / "nope.nii.gz")])
    assert result.exit_code != 0


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "4.0.0" in result.output


def test_parallel_jobs(nifti_dir, tmp_path) -> None:
    image_dir, mask_dir = nifti_dir
    out_dir = tmp_path / "par"
    result = runner.invoke(
        app,
        ["fcm", str(image_dir), "--mask-dir", str(mask_dir), "--output-dir", str(out_dir), "-j", "2"],
    )
    assert result.exit_code == 0, result.output
    assert len(list(out_dir.glob("*_fcm.nii.gz"))) == 3
