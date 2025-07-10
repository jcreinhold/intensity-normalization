# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses `uv` as the modern Python package manager. Key commands:

### Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync --dev
```

### Testing

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=intensity_normalization --cov-report=html

# Run specific test file
uv run pytest tests/test_normalizers.py

# Run specific test
uv run pytest tests/test_normalizers.py::TestFCMNormalizer::test_fcm_basic
```

### Code Quality

```bash
# Format code
uv run ruff format src/intensity_normalization/

# Lint code
uv run ruff check src/intensity_normalization/

# Fix linting issues automatically
uv run ruff check --fix src/intensity_normalization/

# Type checking
uv run mypy src/intensity_normalization/
```

### Building

```bash
# Build package
uv build
```

## Architecture Overview

The codebase follows Clean Architecture principles with clear separation of concerns:

### Domain Layer (`domain/`)

- **protocols.py**: Core interfaces (`ImageProtocol`, `BaseNormalizer`, `PopulationNormalizer`)
- **models.py**: Value objects (`NormalizationConfig`, `Modality`, `TissueType`)
- **exceptions.py**: Domain-specific exceptions

### Adapters Layer (`adapters/`)

- **images.py**: Universal image adapter supporting both numpy arrays and nibabel images
- **io.py**: File I/O operations for loading/saving images

### Normalizers (`normalizers/`)

- **individual/**: Single-image methods (FCM, Z-score, KDE, WhiteStripe)
- **population/**: Multi-image methods (Nyúl, LSQ)

### Services Layer (`services/`)

- **normalization.py**: Orchestration logic via `NormalizationService`
- **validation.py**: Input validation services

### CLI (`cli.py`)

Command-line interface for the `intensity-normalize` command.

## Key Design Patterns

1. **Protocol-Based Design**: Uses Python protocols for flexibility - any object implementing `ImageProtocol` can be normalized.

2. **Service Pattern**: `NormalizationService` orchestrates normalization operations, handling both individual and population methods.

3. **Factory Pattern**: Normalizers are created via registry pattern in `NormalizationService`.

4. **Method Categories**:
   - Individual methods: Fit and transform each image independently
   - Population methods: Fit on multiple images, then transform each

## Adding New Normalizers

1. Create normalizer in appropriate directory (`individual/` or `population/`)
2. Inherit from `BaseNormalizer` or `PopulationNormalizer`
3. Implement `fit()` and `transform()` methods
4. Register in `NORMALIZER_REGISTRY` in `services/normalization.py`
5. Add to exports in `__init__.py`
6. Add tests in `tests/test_normalizers.py`

## Important Implementation Notes

- Population methods (`nyul`, `lsq`) require multiple images for fitting
- The `ImageProtocol` allows seamless support for numpy arrays and nibabel images
- Always preserve image metadata when transforming (use `image.with_data()`)
- Masks are optional but recommended for better normalization results
- The CLI automatically generates output filenames if not specified
