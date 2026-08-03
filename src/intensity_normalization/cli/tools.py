"""Tool subcommands: tissue-membership, plot-histograms, preprocess, coregister."""

from __future__ import annotations

import pathlib
import typing

import nibabel as nib
import nibabel.spatialimages  # explicit so nib.spatialimages resolves
import typer

from intensity_normalization import io


def register(app: typer.Typer) -> None:
    """Attach tool commands to the app."""

    @app.command(name="tissue-membership")
    def tissue_membership_cmd(
        image: pathlib.Path = typer.Argument(..., help="T1-w image."),
        mask: pathlib.Path | None = typer.Option(None, "--mask", "-m", help="Foreground (brain) mask."),
        output: pathlib.Path | None = typer.Option(None, "--output", "-o", help="Output path."),
        hard_segmentation: bool = typer.Option(False, "--hard", help="Hard labels (0/1/2/3) instead of memberships."),
        seed: int = typer.Option(0, "--seed", help="RNG seed for the fuzzy c-means fit."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
    ) -> None:
        """Fuzzy c-means CSF/GM/WM membership maps of a T1-w image."""
        import intensity_normalization as inorm

        img = io.load_image(image)
        msk = io.load_image(mask) if mask else None
        memberships = inorm.tissue_membership(img, msk, hard_segmentation=hard_segmentation, seed=seed)
        out = output or io.output_path(image, suffix="tissue_membership")
        io.save_image(typing.cast(nib.spatialimages.SpatialImage, memberships), out)
        if not quiet:
            typer.echo(f"wrote {out}")

    @app.command(name="plot-histograms")
    def plot_histograms_cmd(
        image_dir: pathlib.Path = typer.Argument(..., help="Directory of images."),
        mask_dir: pathlib.Path | None = typer.Option(None, "--mask-dir", "-m", help="Masks matched by filename."),
        output: pathlib.Path | None = typer.Option(
            None, "--output", "-o", help="Save the figure here (shows it otherwise)."
        ),
        linear: bool = typer.Option(False, "--linear", help="Linear density axis (default is log)."),
        seed: int = typer.Option(0, "--seed", help="RNG seed for the KDE subsample."),
    ) -> None:
        """Plot smoothed foreground histograms of a set of images.

        The recommended way to validate normalization results: plot before
        and after, and compare.
        """
        import intensity_normalization as inorm

        paths = io.find_images(image_dir, exclude=["tissue_membership"])
        mask_paths = io.match_masks(paths, mask_dir) if mask_dir else None
        images = [io.load_image(p) for p in paths]
        masks = [io.load_image(m) for m in mask_paths] if mask_paths else None
        labels = [io.split_filename(p)[1] for p in paths]
        inorm.plot_histograms(images, masks, labels=labels, log_scale=not linear, output=output, seed=seed)
        if output is not None:
            typer.echo(f"wrote {output}")
        else:
            import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]  # optional dep

            plt.show()

    @app.command()
    def preprocess(
        image: pathlib.Path = typer.Argument(..., help="Image to preprocess."),
        mask: pathlib.Path | None = typer.Option(None, "--mask", "-m", help="Foreground (brain) mask."),
        output: pathlib.Path | None = typer.Option(None, "--output", "-o", help="Output path."),
        mask_output: pathlib.Path | None = typer.Option(None, "--mask-output", help="Output path for the mask."),
        resolution: list[float] | None = typer.Option(None, "--resolution", "-r", help="Resample to X Y Z mm."),
        orientation: str = typer.Option("RAS", "--orientation", help="Reorient to this ANTs code."),
        single_n4: bool = typer.Option(False, "--single-n4", help="Skip the second (smoothed-mask) N4 pass."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
    ) -> None:
        """N4 bias-field correction + optional resample/reorient (requires [ants])."""
        from rich.progress import Progress, SpinnerColumn, TextColumn

        import intensity_normalization as inorm

        img = io.load_image(image)
        msk = io.load_image(mask) if mask else None
        res: tuple[float, float, float] | None = None
        if resolution:
            if len(resolution) != 3:
                raise typer.BadParameter("--resolution takes exactly three values: X Y Z")
            res = (resolution[0], resolution[1], resolution[2])
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), disable=quiet) as progress:
            progress.add_task("preprocessing (N4 bias correction)", total=None)
            out_img, out_mask = inorm.preprocess(
                img,
                msk,
                resolution=res,
                orientation=orientation,
                second_n4_with_smoothed_mask=not single_n4,
            )
        out = io.save_image(
            typing.cast(nib.spatialimages.SpatialImage, out_img), output or io.output_path(image, suffix="preprocessed")
        )
        out_mask_path = mask_output or io.output_path(image, suffix="mask")
        io.save_image(typing.cast(nib.spatialimages.SpatialImage, out_mask), out_mask_path)
        if not quiet:
            typer.echo(f"wrote {out}")
            typer.echo(f"wrote {out_mask_path}")

    @app.command()
    def coregister(
        images: list[pathlib.Path] = typer.Argument(..., help="Images to register."),
        template: pathlib.Path | None = typer.Option(None, "--template", "-t", help="Target image (MNI if omitted)."),
        output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory."),
        type_of_transform: str = typer.Option("Affine", "--type-of-transform", help="Rigid, Affine, SyN, ..."),
        template_mask: pathlib.Path | None = typer.Option(None, "--template-mask", help="Mask on the template."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
    ) -> None:
        """Co-register images to a template with ANTs (requires [ants])."""
        from rich.progress import track

        import intensity_normalization as inorm

        tpl = io.load_image(template) if template else None
        tpl_mask = io.load_image(template_mask) if template_mask else None
        for path in track(images, description="coregister", disable=quiet):
            img = io.load_image(path)
            registered = inorm.coregister(img, tpl, type_of_transform=type_of_transform, template_mask=tpl_mask)
            out = io.save_image(
                typing.cast(nib.spatialimages.SpatialImage, registered),
                io.output_path(path, suffix="registered", output_dir=output_dir),
            )
            if not quiet:
                typer.echo(f"wrote {out}")
