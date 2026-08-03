"""Command-line interface: ``intensity-normalize <command>``.

The only stringly-typed layer in the package. Commands mirror the Python API:
individual methods take one or more images (or a directory), population methods
take a directory, tools take their natural inputs.
"""

from __future__ import annotations

import sys

import typer

from intensity_normalization import __version__
from intensity_normalization.errors import IntensityNormalizationError

__all__ = ["app", "main"]

app = typer.Typer(
    name="intensity-normalize",
    help="Normalize the intensities of MR images.",
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"intensity-normalization {__version__}")
        raise typer.Exit


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show full tracebacks on errors.",
    ),
) -> None:
    """Normalize the intensities of MR images.

    Individual methods (zscore, fcm, kde, whitestripe) normalize each image
    independently. Population methods (nyul, lsq, ravel) learn a transform
    from a set of images. See the per-command help for details.
    """


def main() -> None:
    """Entry point; converts domain errors into clean CLI errors."""
    from intensity_normalization.cli import normalize, tools

    normalize.register(app)
    tools.register(app)
    try:
        app()
    except IntensityNormalizationError as exn:
        if "--debug" in sys.argv:
            raise
        from rich.console import Console

        Console(stderr=True).print(f"[bold red]error:[/bold red] {exn}")
        sys.exit(1)
