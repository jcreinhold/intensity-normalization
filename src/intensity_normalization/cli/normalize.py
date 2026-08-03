"""Normalization subcommands: the 7 methods, individual and population.

The shared option vocabulary (``ImagesArg``, ``MaskOpt``, ...) is defined once
at the top of this module so the commands cannot drift apart; each command
function is a thin dispatcher mapping CLI options onto the Python API.
"""

from __future__ import annotations

import pathlib
import typing
from concurrent.futures import ProcessPoolExecutor
from typing import Annotated

import nibabel as nib
import nibabel.spatialimages  # explicit so nib.spatialimages resolves
import numpy as np
import typer

from intensity_normalization import io
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods._transform import FittedTransform

MethodKwargs = dict[str, typing.Any]
Job = dict[str, typing.Any]

# --- the shared CLI contract: each option defined exactly once ---------------

ImagesArg = Annotated[list[pathlib.Path], typer.Argument(help="Image(s) or directories to normalize.")]
MaskOpt = Annotated[pathlib.Path | None, typer.Option("--mask", "-m", help="Foreground (brain) mask (single image).")]
MaskDirOpt = Annotated[pathlib.Path | None, typer.Option("--mask-dir", help="Masks matched by filename (batch).")]
OutputOpt = Annotated[pathlib.Path | None, typer.Option("--output", "-o", help="Output path (single image).")]
OutputDirOpt = Annotated[pathlib.Path | None, typer.Option("--output-dir", help="Output directory (batch).")]
NormValueOpt = Annotated[float, typer.Option("--norm-value", help="Intensity the reference value maps to.")]
ModalityOpt = Annotated[str, typer.Option("--modality", "-mo", help="MR modality (t1, t2, flair, pd, md, other).")]
PeakOpt = Annotated[str | None, typer.Option("--peak", help="Explicit tissue peak (last, largest, first).")]
SeedOpt = Annotated[int, typer.Option("--seed", help="RNG seed (deterministic).")]
JobsOpt = Annotated[int, typer.Option("--jobs", "-j", help="Parallel workers for batches.")]
PlotOpt = Annotated[bool, typer.Option("--plot", "-p", help="Plot foreground histograms before/after.")]
QuietOpt = Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress output.")]

ImageDirArg = Annotated[
    pathlib.Path, typer.Argument(help="Directory of images (fit set, or inputs with --load-state).")
]
PopMaskDirOpt = Annotated[pathlib.Path | None, typer.Option("--mask-dir", "-m", help="Masks matched by filename.")]
PopOutputDirOpt = Annotated[pathlib.Path | None, typer.Option("--output-dir", "-o", help="Output directory.")]
SaveStateOpt = Annotated[pathlib.Path | None, typer.Option("--save-state", help="Save the fitted transform (.npz).")]
LoadStateOpt = Annotated[
    pathlib.Path | None, typer.Option("--load-state", help="Apply a saved transform instead of fitting.")
]

# --- batch plumbing ------------------------------------------------------------


def _method_fn(name: str) -> typing.Callable[..., typing.Any]:
    import intensity_normalization as inorm

    return {
        "zscore": inorm.zscore,
        "fcm": inorm.fcm,
        "kde": inorm.kde,
        "whitestripe": inorm.whitestripe,
    }[name]


def _normalize_one(job: Job) -> pathlib.Path:
    """Module-level worker: load one image, normalize, save (process-pool safe)."""
    image = io.load_image(job["image"])
    mask = io.load_image(job["mask"]) if job["mask"] else None
    normalized = _method_fn(job["method"])(image, mask, **job["kwargs"])
    return io.save_image(normalized, job["output"])


def _resolve_inputs(inputs: list[pathlib.Path]) -> list[pathlib.Path]:
    """Expand directories to their images; validate single files exist."""
    paths: list[pathlib.Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(io.find_images(item))
        elif item.is_file():
            paths.append(item)
        else:
            raise IntensityNormalizationError(f"No such file or directory: {item}")
    return paths


def _resolve_masks(
    inputs: list[pathlib.Path],
    mask: pathlib.Path | None,
    mask_dir: pathlib.Path | None,
) -> list[pathlib.Path | None]:
    if mask is not None and mask_dir is not None:
        raise IntensityNormalizationError("Pass either --mask or --mask-dir, not both.")
    if mask is not None:
        if len(inputs) != 1:
            raise IntensityNormalizationError("--mask works with a single image; use --mask-dir for batches.")
        if not mask.is_file():
            raise IntensityNormalizationError(f"Mask not found: {mask}")
        return [mask]
    if mask_dir is not None:
        return list(io.match_masks(inputs, mask_dir))
    return [None] * len(inputs)


def _resolve_outputs(
    inputs: list[pathlib.Path],
    output: pathlib.Path | None,
    output_dir: pathlib.Path | None,
    suffix: str,
) -> list[pathlib.Path]:
    if output is not None and output_dir is not None:
        raise IntensityNormalizationError("Pass either --output or --output-dir, not both.")
    if output is not None:
        if len(inputs) != 1:
            raise IntensityNormalizationError("--output works with a single image; use --output-dir for batches.")
        return [output]
    return [io.output_path(p, suffix=suffix, output_dir=output_dir) for p in inputs]


def _run_individual(
    method: str,
    inputs: list[pathlib.Path],
    *,
    mask: pathlib.Path | None,
    mask_dir: pathlib.Path | None,
    output: pathlib.Path | None,
    output_dir: pathlib.Path | None,
    jobs: int,
    quiet: bool,
    plot: bool,
    kwargs: MethodKwargs,
) -> None:
    from rich.progress import track

    paths = _resolve_inputs(inputs)
    mask_paths = _resolve_masks(paths, mask, mask_dir)
    output_paths = _resolve_outputs(paths, output, output_dir, method)
    job_list: list[Job] = [
        {"method": method, "image": p, "mask": m, "output": o, "kwargs": kwargs}
        for p, m, o in zip(paths, mask_paths, output_paths, strict=True)
    ]

    if jobs > 1 and len(job_list) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_normalize_one, job) for job in job_list]
            done = list(track((f.result() for f in futures), total=len(futures), description=method, disable=quiet))
    else:
        done = [_normalize_one(job) for job in track(job_list, description=method, disable=quiet or len(job_list) == 1)]

    if not quiet:
        for path in done:
            typer.echo(f"wrote {path}")

    if plot:
        import matplotlib.pyplot as plt

        import intensity_normalization as inorm

        before = [io.load_image(p) for p in paths]
        after = [io.load_image(p) for p in done]
        masks_loaded = [io.load_image(m) if m else None for m in mask_paths]
        inorm.plot_histograms(before, masks_loaded, title=f"before {method}")
        inorm.plot_histograms(after, masks_loaded, title=f"after {method}")
        plt.show()


def _run_population(
    method: str,
    module: typing.Any,
    transform_cls: type[FittedTransform],
    image_dir: pathlib.Path,
    *,
    mask_dir: pathlib.Path | None,
    output_dir: pathlib.Path | None,
    save_state: pathlib.Path | None,
    load_state: pathlib.Path | None,
    quiet: bool,
    fit_kwargs: MethodKwargs | None = None,
    save_tissue_maps: bool = False,
) -> None:
    from rich.progress import Progress, SpinnerColumn, TextColumn, track

    paths = io.find_images(image_dir)
    mask_paths = list(io.match_masks(paths, mask_dir)) if mask_dir else None

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), disable=quiet) as progress:
        progress.add_task("loading images", total=None)
        images = [io.load_image(p) for p in paths]
        masks = [io.load_image(m) for m in mask_paths] if mask_paths else None

    if load_state is not None:
        tx = transform_cls.load(load_state)
    else:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), disable=quiet) as progress:
            progress.add_task(f"fitting {method}", total=None)
            tx = module.fit(images, masks, **(fit_kwargs or {}))
        if save_state is not None:
            tx.save(save_state)
            if not quiet:
                typer.echo(f"saved transform to {save_state}")

    if save_tissue_maps:
        # lsq persists the reference membership map in the transform, so this
        # works identically after --load-state
        membership = typing.cast(typing.Any, tx).reference_membership
        affine = images[0].affine if images[0].shape == membership.shape[:3] else np.eye(4)
        ref_out = io.output_path(paths[0], suffix="tissue_membership", output_dir=output_dir)
        io.save_image(nib.nifti1.Nifti1Image(membership, affine), ref_out)
        if not quiet:
            typer.echo(f"wrote {ref_out}")

    for i, path in enumerate(track(paths, description=f"applying {method}", disable=quiet)):
        mask = masks[i] if masks else None
        normalized = tx(images[i], mask)
        out = io.save_image(
            typing.cast("nib.spatialimages.SpatialImage", normalized),
            io.output_path(path, suffix=method, output_dir=output_dir),
        )
        if not quiet:
            typer.echo(f"wrote {out}")


# --- commands -------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    """Attach normalization commands to the app."""

    @app.command()
    def zscore(
        images: ImagesArg,
        mask: MaskOpt = None,
        mask_dir: MaskDirOpt = None,
        output: OutputOpt = None,
        output_dir: OutputDirOpt = None,
        norm_value: NormValueOpt = 1.0,
        jobs: JobsOpt = 1,
        plot: PlotOpt = False,
        quiet: QuietOpt = False,
    ) -> None:
        """Standardize each image by its foreground mean and standard deviation."""
        _run_individual(
            "zscore",
            images,
            mask=mask,
            mask_dir=mask_dir,
            output=output,
            output_dir=output_dir,
            jobs=jobs,
            quiet=quiet,
            plot=plot,
            kwargs={"norm_value": norm_value},
        )

    @app.command()
    def fcm(
        images: ImagesArg,
        mask: MaskOpt = None,
        mask_dir: MaskDirOpt = None,
        output: OutputOpt = None,
        output_dir: OutputDirOpt = None,
        tissue: Annotated[str, typer.Option("--tissue", "-t", help="Tissue class (csf, gm, wm).")] = "wm",
        norm_value: NormValueOpt = 1.0,
        seed: SeedOpt = 0,
        jobs: JobsOpt = 1,
        plot: PlotOpt = False,
        quiet: QuietOpt = False,
    ) -> None:
        """Normalize each T1-w image to the fuzzy c-means mean of a tissue class.

        Recommended starting point for T1-w brain images.
        """
        _run_individual(
            "fcm",
            images,
            mask=mask,
            mask_dir=mask_dir,
            output=output,
            output_dir=output_dir,
            jobs=jobs,
            quiet=quiet,
            plot=plot,
            kwargs={"tissue": tissue, "norm_value": norm_value, "seed": seed},
        )

    @app.command()
    def kde(
        images: ImagesArg,
        mask: MaskOpt = None,
        mask_dir: MaskDirOpt = None,
        output: OutputOpt = None,
        output_dir: OutputDirOpt = None,
        modality: ModalityOpt = "t1",
        peak: PeakOpt = None,
        norm_value: NormValueOpt = 1.0,
        seed: SeedOpt = 0,
        jobs: JobsOpt = 1,
        plot: PlotOpt = False,
        quiet: QuietOpt = False,
    ) -> None:
        """Normalize each image by the tissue mode of its smoothed histogram."""
        _run_individual(
            "kde",
            images,
            mask=mask,
            mask_dir=mask_dir,
            output=output,
            output_dir=output_dir,
            jobs=jobs,
            quiet=quiet,
            plot=plot,
            kwargs={"modality": modality, "peak": peak, "norm_value": norm_value, "seed": seed},
        )

    @app.command()
    def whitestripe(
        images: ImagesArg,
        mask: MaskOpt = None,
        mask_dir: MaskDirOpt = None,
        output: OutputOpt = None,
        output_dir: OutputDirOpt = None,
        modality: ModalityOpt = "t1",
        peak: PeakOpt = None,
        width: Annotated[float, typer.Option("--width", help="Quantile half-width of the white stripe.")] = 0.05,
        norm_value: NormValueOpt = 1.0,
        seed: SeedOpt = 0,
        jobs: JobsOpt = 1,
        plot: PlotOpt = False,
        quiet: QuietOpt = False,
    ) -> None:
        """WhiteStripe: standardize by the normal-appearing white matter."""
        _run_individual(
            "whitestripe",
            images,
            mask=mask,
            mask_dir=mask_dir,
            output=output,
            output_dir=output_dir,
            jobs=jobs,
            quiet=quiet,
            plot=plot,
            kwargs={"modality": modality, "peak": peak, "width": width, "norm_value": norm_value, "seed": seed},
        )

    @app.command()
    def nyul(
        image_dir: ImageDirArg,
        mask_dir: PopMaskDirOpt = None,
        output_dir: PopOutputDirOpt = None,
        save_state: SaveStateOpt = None,
        load_state: LoadStateOpt = None,
        quiet: QuietOpt = False,
    ) -> None:
        """Nyúl & Udupa piecewise-linear histogram matching over a set of images.

        Fits the standard histogram on IMAGE_DIR, normalizes the images, and
        writes <name>_nyul.nii.gz. With --load-state, applies a previously
        saved transform to the images instead of fitting.
        """
        from intensity_normalization.methods import nyul as nyul_mod
        from intensity_normalization.methods.nyul import NyulTransform

        _run_population(
            "nyul",
            nyul_mod,
            NyulTransform,
            image_dir,
            mask_dir=mask_dir,
            output_dir=output_dir,
            save_state=save_state,
            load_state=load_state,
            quiet=quiet,
        )

    @app.command()
    def lsq(
        image_dir: ImageDirArg,
        mask_dir: PopMaskDirOpt = None,
        output_dir: PopOutputDirOpt = None,
        save_state: SaveStateOpt = None,
        load_state: LoadStateOpt = None,
        norm_value: NormValueOpt = 1.0,
        seed: SeedOpt = 0,
        save_tissue_maps: Annotated[
            bool, typer.Option("--save-tissue-maps", help="Save the reference image's tissue membership map.")
        ] = False,
        quiet: QuietOpt = False,
    ) -> None:
        """Least-squares scaling of CSF/GM/WM tissue means across a set of images."""
        from intensity_normalization.methods import lsq as lsq_mod
        from intensity_normalization.methods.lsq import LSQTransform

        _run_population(
            "lsq",
            lsq_mod,
            LSQTransform,
            image_dir,
            mask_dir=mask_dir,
            output_dir=output_dir,
            save_state=save_state,
            load_state=load_state,
            quiet=quiet,
            fit_kwargs={"norm_value": norm_value, "seed": seed},
            save_tissue_maps=save_tissue_maps,
        )

    @app.command()
    def ravel(
        image_dir: Annotated[pathlib.Path, typer.Argument(help="Directory of co-registered images.")],
        mask_dir: PopMaskDirOpt = None,
        output_dir: PopOutputDirOpt = None,
        save_state: Annotated[
            pathlib.Path | None, typer.Option("--save-state", help="Save the learned artifacts (.npz).")
        ] = None,
        num_unwanted_factors: Annotated[
            int, typer.Option("-b", "--num-unwanted-factors", help="Unwanted factors to remove.")
        ] = 1,
        membership_threshold: Annotated[
            float, typer.Option("--membership-threshold", help="FCM CSF membership threshold.")
        ] = 0.99,
        no_registration: Annotated[
            bool, typer.Option("--no-registration", help="Images are already deformably co-registered.")
        ] = False,
        sparse_svd: Annotated[
            bool, typer.Option("--sparse-svd", help="Lower-memory SVD for factor estimation.")
        ] = False,
        masks_are_csf: Annotated[
            bool, typer.Option("--masks-are-csf", help="Masks are CSF masks, not brain masks.")
        ] = False,
        quantile_to_label_csf: Annotated[
            float, typer.Option("--quantile-to-label-csf", help="Fraction of images a voxel must be CSF in.")
        ] = 1.0,
        seed: SeedOpt = 0,
        quiet: QuietOpt = False,
    ) -> None:
        """RAVEL: WhiteStripe + CSF control-voxel correction of a co-registered set.

        All images must share shape and voxel correspondence; co-register them
        first (see the coregister command) or use --no-registration.
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn

        import intensity_normalization as inorm

        paths = io.find_images(image_dir)
        mask_paths = io.match_masks(paths, mask_dir) if mask_dir else None
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), disable=quiet) as progress:
            progress.add_task("loading images", total=None)
            images = [io.load_image(p) for p in paths]
            masks = [io.load_image(m) for m in mask_paths] if mask_paths else None
            progress.add_task("RAVEL (WhiteStripe, registration, correction)", total=None)
            result, normalized = inorm.ravel.fit_transform(
                images,
                masks,
                register=not no_registration,
                membership_threshold=membership_threshold,
                num_unwanted_factors=num_unwanted_factors,
                sparse_svd=sparse_svd,
                quantile_to_label_csf=quantile_to_label_csf,
                masks_are_csf=masks_are_csf,
                seed=seed,
            )
        for path, normed in zip(paths, normalized, strict=True):
            out = io.save_image(
                typing.cast(nib.spatialimages.SpatialImage, normed),
                io.output_path(path, suffix="ravel", output_dir=output_dir),
            )
            if not quiet:
                typer.echo(f"wrote {out}")
        if save_state is not None:
            result.save(save_state)
            if not quiet:
                typer.echo(f"saved artifacts to {save_state}")
