"""Unified command-line interface for intensity normalization."""

import argparse
import sys
from pathlib import Path

from intensity_normalization.adapters.io import generate_output_path, load_image, save_image, validate_input_path
from intensity_normalization.domain.exceptions import IntensityNormalizationError
from intensity_normalization.domain.models import Modality, NormalizationConfig, TissueType
from intensity_normalization.services.normalization import NormalizationService


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Validate input file
        input_path = validate_input_path(args.input)

        # Load images
        image = load_image(input_path)
        mask = load_image(args.mask) if args.mask else None

        # Create configuration
        config = NormalizationConfig(
            method=args.method,
            modality=Modality(args.modality),
            tissue_type=TissueType(args.tissue_type),
        )

        # Normalize
        normalized = NormalizationService.normalize_image(image, config, mask)

        # Determine output path
        output_path = Path(args.output) if args.output else generate_output_path(input_path, args.method)

        # Save result
        save_image(normalized, output_path)

        print(f"Normalized image saved to: {output_path}")
        return 0

    except IntensityNormalizationError as e:
        print(f"Normalization error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Normalize MR image intensities using various methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "method",
        choices=NormalizationService.get_available_methods(),
        help="Normalization method to use",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input image path (supports: .nii, .nii.gz, .mgz, .mnc, etc.)",
    )

    # Optional arguments
    parser.add_argument(
        "-m",
        "--mask",
        type=Path,
        help="Brain mask image path (optional)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (defaults to input_<method>.ext)",
    )

    parser.add_argument(
        "--modality",
        choices=[m.value for m in Modality],
        default="t1",
        help="MR image modality",
    )

    parser.add_argument(
        "--tissue-type",
        choices=[t.value for t in TissueType],
        default="wm",
        help="Target tissue type for tissue-based methods",
    )

    # Method-specific arguments (would be extended for each method)
    method_group = parser.add_argument_group("method-specific arguments")

    method_group.add_argument(
        "--width",
        type=float,
        default=0.05,
        help="Width parameter for WhiteStripe method",
    )

    method_group.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters for FCM method",
    )

    # Utility arguments
    parser.add_argument(
        "--version",
        action="version",
        version="intensity-normalization 3.0.0",
        help="Show version information",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def normalize_image_cli() -> int:
    """Convenience entry point for single image normalization."""
    return main()


def normalize_directory_cli() -> int:
    """Entry point for directory-based normalization (population methods)."""
    parser = create_directory_parser()
    args = parser.parse_args()

    try:
        # Find images in directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
            return 1

        # Get image files
        extensions = ["*.nii", "*.nii.gz", "*.mgz", "*.mgh", "*.mnc"]
        image_files: list[Path] = []
        for ext in extensions:
            image_files.extend(input_dir.glob(ext))

        if not image_files:
            print(f"No supported image files found in: {input_dir}", file=sys.stderr)
            return 1

        # Load images
        images = [load_image(img_file) for img_file in image_files]

        # Load masks if provided
        masks = None
        if args.mask_dir:
            mask_dir = Path(args.mask_dir)
            mask_files: list[Path | None] = []
            for img_file in image_files:
                mask_file = mask_dir / img_file.name
                if mask_file.exists():
                    mask_files.append(mask_file)
                else:
                    mask_files.append(None)
            masks = [load_image(mask_file) if mask_file else None for mask_file in mask_files]

        # Create configuration
        config = NormalizationConfig(
            method=args.method,
            modality=Modality(args.modality),
            tissue_type=TissueType(args.tissue_type),
        )

        # Normalize
        normalized_images = NormalizationService.normalize_images(images, config, masks)

        # Save results
        output_dir = Path(args.output_dir) if args.output_dir else input_dir
        output_dir.mkdir(exist_ok=True)

        for img_file, normalized in zip(image_files, normalized_images, strict=False):
            output_path = generate_output_path(output_dir / img_file.name, args.method)
            save_image(normalized, output_path)
            if args.verbose:
                print(f"Saved: {output_path}")

        print(f"Normalized {len(normalized_images)} images")
        return 0

    except IntensityNormalizationError as e:
        print(f"Normalization error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def create_directory_parser() -> argparse.ArgumentParser:
    """Create CLI parser for directory-based normalization."""
    parser = argparse.ArgumentParser(
        description="Normalize multiple MR images using population-based methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "method",
        choices=[
            m for m in NormalizationService.get_available_methods() if NormalizationService.is_population_method(m)
        ],
        help="Population-based normalization method",
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing input images",
    )

    # Optional arguments
    parser.add_argument(
        "-m",
        "--mask-dir",
        type=Path,
        help="Directory containing brain masks (optional)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (defaults to input directory)",
    )

    parser.add_argument(
        "--modality",
        choices=[m.value for m in Modality],
        default="t1",
        help="MR image modality",
    )

    parser.add_argument(
        "--tissue-type",
        choices=[t.value for t in TissueType],
        default="wm",
        help="Target tissue type for tissue-based methods",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="intensity-normalization 3.0.0",
        help="Show version information",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


if __name__ == "__main__":
    sys.exit(main())
