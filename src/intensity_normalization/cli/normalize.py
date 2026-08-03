"""Normalization subcommands: the 7 methods, individual and population."""

from __future__ import annotations

import pathlib
import typing
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import typer

from intensity_normalization import io
from intensity_normalization.errors import IntensityNormalizationError
from intensity_normalization.methods._transform import FittedTransform

MethodKwargs = dict[str, typing.Any]
Job = dict[str, typing.Any]

_INDIVIDUAL_METHODS = ("zscore", "fcm", "kde", "whitestripe")


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
    paths = _resolve_inputs(inputs)
    mask_paths = _resolve_masks(paths, mask, mask_dir)
    output_paths = _resolve_outputs(paths, output, output_dir, method)
    job_list: list[Job] = [
        {"method": method, "image": p, "mask": m, "output": o, "kwargs": kwargs}
        for p, m, o in zip(paths, mask_paths, output_paths, strict=True)
    ]

    if jobs > 1 and len(job_list) > 1:
        from rich.progress import track

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_normalize_one, job) for job in job_list]
            iterator = (f.result() for f in futures)
            done = list(track(iterator, total=len(futures), description=method, disable=quiet))
    else:
        from rich.progress import track

        done = [_normalize_one(job) for job in track(job_list, description=method, disable=quiet or len(job_list) == 1)]

    if not quiet:
        for path in done:
            typer.echo(f"wrote {path}")

    if plot:
        import intensity_normalization as inorm

        before = [io.load_image(p) for p in paths]
        after = [io.load_image(p) for p in done]
        masks_loaded = [io.load_image(m) if m else None for m in mask_paths]
        inorm.plot_histograms(before, masks_loaded, title=f"before {method}")
        inorm.plot_histograms(after, masks_loaded, title=f"after {method}")
        import matplotlib.pyplot as plt

        plt.show()


def register(app: typer.Typer) -> None:
    """Attach normalization commands to the app."""

    @app.command()
    def zscore(
        images: list[pathlib.Path] = typer.Argument(..., help="Image(s) or directories to normalize."),
        mask: pathlib.Path | None = typer.Option(None, "--mask", "-m", help="Foreground (brain) mask (single image)."),
        mask_dir: pathlib.Path | None = typer.Option(None, "--mask-dir", help="Masks matched by filename (batch)."),
        output: pathlib.Path | None = typer.Option(None, "--output", "-o", help="Output path (single image)."),
        output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", help="Output directory (batch)."),
        norm_value: float = typer.Option(1.0, "--norm-value", help="Scale the standardized image by this."),
        jobs: int = typer.Option(1, "--jobs", "-j", help="Parallel workers for batches."),
        plot: bool = typer.Option(False, "--plot", "-p", help="Plot foreground histograms before/after."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
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
        images: list[pathlib.Path] = typer.Argument(..., help="Image(s) or directories to normalize."),
        mask: pathlib.Path | None = typer.Option(None, "--mask", "-m", help="Foreground (brain) mask (single image)."),
        mask_dir: pathlib.Path | None = typer.Option(None, "--mask-dir", help="Masks matched by filename (batch)."),
        output: pathlib.Path | None = typer.Option(None, "--output", "-o", help="Output path (single image)."),
        output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", help="Output directory (batch)."),
        tissue: str = typer.Option("wm", "--tissue", "-t", help="Tissue class (csf, gm, wm)."),
        norm_value: float = typer.Option(1.0, "--norm-value", help="Intensity the tissue mean maps to."),
        seed: int = typer.Option(0, "--seed", help="RNG seed for the fuzzy c-means fit."),
        jobs: int = typer.Option(1, "--jobs", "-j", help="Parallel workers for batches."),
        plot: bool = typer.Option(False, "--plot", "-p", help="Plot foreground histograms before/after."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
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
        images: list[pathlib.Path] = typer.Argument(..., help="Image(s) or directories to normalize."),
        mask: pathlib.Path | None = typer.Option(None, "--mask", "-m", help="Foreground (brain) mask (single image)."),
        mask_dir: pathlib.Path | None = typer.Option(None, "--mask-dir", help="Masks matched by filename (batch)."),
        output: pathlib.Path | None = typer.Option(None, "--output", "-o", help="Output path (single image)."),
        output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", help="Output directory (batch)."),
        modality: str = typer.Option("t1", "--modality", "-mo", help="MR modality (t1, t2, flair, pd, md, other)."),
        peak: str | None = typer.Option(None, "--peak", help="Explicit tissue peak (last, largest, first)."),
        norm_value: float = typer.Option(1.0, "--norm-value", help="Intensity the tissue mode maps to."),
        seed: int = typer.Option(0, "--seed", help="RNG seed for the KDE subsample."),
        jobs: int = typer.Option(1, "--jobs", "-j", help="Parallel workers for batches."),
        plot: bool = typer.Option(False, "--plot", "-p", help="Plot foreground histograms before/after."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
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
        images: list[pathlib.Path] = typer.Argument(..., help="Image(s) or directories to normalize."),
        mask: pathlib.Path | None = typer.Option(None, "--mask", "-m", help="Foreground (brain) mask (single image)."),
        mask_dir: pathlib.Path | None = typer.Option(None, "--mask-dir", help="Masks matched by filename (batch)."),
        output: pathlib.Path | None = typer.Option(None, "--output", "-o", help="Output path (single image)."),
        output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", help="Output directory (batch)."),
        modality: str = typer.Option("t1", "--modality", "-mo", help="MR modality (t1, t2, flair, pd, md, other)."),
        peak: str | None = typer.Option(None, "--peak", help="Explicit tissue peak (last, largest, first)."),
        width: float = typer.Option(0.05, "--width", help="Quantile half-width of the white stripe."),
        norm_value: float = typer.Option(1.0, "--norm-value", help="Scale the standardized image by this."),
        seed: int = typer.Option(0, "--seed", help="RNG seed for the KDE subsample."),
        jobs: int = typer.Option(1, "--jobs", "-j", help="Parallel workers for batches."),
        plot: bool = typer.Option(False, "--plot", "-p", help="Plot foreground histograms before/after."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
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
        image_dir: pathlib.Path = typer.Argument(
            ..., help="Directory of images (fit set, or inputs with --load-state)."
        ),
        mask_dir: pathlib.Path | None = typer.Option(None, "--mask-dir", "-m", help="Masks matched by filename."),
        output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory."),
        save_state: pathlib.Path | None = typer.Option(None, "--save-state", help="Save the fitted transform (.npz)."),
        load_state: pathlib.Path | None = typer.Option(
            None, "--load-state", help="Apply a saved transform instead of fitting."
        ),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
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
        image_dir: pathlib.Path = typer.Argument(
            ..., help="Directory of T1-w images (fit set, or inputs with --load-state)."
        ),
        mask_dir: pathlib.Path | None = typer.Option(None, "--mask-dir", "-m", help="Masks matched by filename."),
        output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory."),
        save_state: pathlib.Path | None = typer.Option(None, "--save-state", help="Save the fitted transform (.npz)."),
        load_state: pathlib.Path | None = typer.Option(
            None, "--load-state", help="Apply a saved transform instead of fitting."
        ),
        norm_value: float = typer.Option(1.0, "--norm-value", help="Intensity the reference CSF mean maps to."),
        seed: int = typer.Option(0, "--seed", help="RNG seed for the fuzzy c-means fits."),
        save_tissue_maps: bool = typer.Option(
            False, "--save-tissue-maps", help="Save the reference image's tissue membership map."
        ),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
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
            fit_kwargs={"norm_value": norm_value, "seed": seed, "return_tissue_maps": save_tissue_maps},
            save_tissue_maps=save_tissue_maps,
        )

    @app.command()
    def ravel(
        image_dir: pathlib.Path = typer.Argument(..., help="Directory of co-registered images."),
        mask_dir: pathlib.Path | None = typer.Option(None, "--mask-dir", "-m", help="Masks matched by filename."),
        output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory."),
        save_state: pathlib.Path | None = typer.Option(None, "--save-state", help="Save the learned artifacts (.npz)."),
        num_unwanted_factors: int = typer.Option(1, "-b", "--num-unwanted-factors", help="Unwanted factors to remove."),
        membership_threshold: float = typer.Option(
            0.99, "--membership-threshold", help="FCM CSF membership threshold."
        ),
        no_registration: bool = typer.Option(
            False, "--no-registration", help="Images are already deformably co-registered."
        ),
        sparse_svd: bool = typer.Option(False, "--sparse-svd", help="Lower-memory SVD for factor estimation."),
        masks_are_csf: bool = typer.Option(False, "--masks-are-csf", help="Masks are CSF masks, not brain masks."),
        quantile_to_label_csf: float = typer.Option(
            1.0, "--quantile-to-label-csf", help="Fraction of images a voxel must be CSF in."
        ),
        seed: int = typer.Option(0, "--seed", help="RNG seed for the tissue fits."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
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
            fit_out = module.fit(images, masks, **(fit_kwargs or {}))
        if save_tissue_maps:
            tx, tissue_map = fit_out
            ref_out = io.output_path(paths[0], suffix="tissue_membership", output_dir=output_dir)
            import nibabel as nib

            ref_image = images[0]
            tm_img = nib.nifti1.Nifti1Image(tissue_map, ref_image.affine, ref_image.header)
            io.save_image(tm_img, ref_out)
            if not quiet:
                typer.echo(f"wrote {ref_out}")
        else:
            tx = fit_out
        if save_state is not None:
            tx.save(save_state)
            if not quiet:
                typer.echo(f"saved transform to {save_state}")

    for i, path in enumerate(track(paths, description=f"applying {method}", disable=quiet)):
        mask = masks[i] if masks else None
        normalized = tx(images[i], mask)
        out = io.save_image(
            typing.cast("nib.spatialimages.SpatialImage", normalized),
            io.output_path(path, suffix=method, output_dir=output_dir),
        )
        if not quiet:
            typer.echo(f"wrote {out}")
