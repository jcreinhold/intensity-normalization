# Intensity Normalization

A modern Python package for normalizing MR image intensities.

## Features

- **🔧 Multiple Image Format Support**: Works with numpy arrays and nibabel images (.nii, .nii.gz, .mgz, .mnc, etc.)
- **📊 6 Normalization Methods**: FCM, KDE, WhiteStripe, Z-score, Nyúl, LSQ
- **⚡ High Performance**: Optimized implementations

## Installation

```bash
pip install intensity-normalization
```

## Quick Start

### High-Level API

```python
import numpy as np
import nibabel as nib
from intensity_normalization import normalize_image

# Load MRI image and brain mask
img = nib.load("brain_t1.nii.gz")
mask = nib.load("brain_mask.nii.gz")

# Normalize using FCM (default, recommended for T1)
normalized = normalize_image(img, method="fcm", mask=mask)

# Different methods for different modalities
t2_normalized = normalize_image(img, method="kde", modality="t2")
zscore_normalized = normalize_image(img, method="zscore")
```

### Object-Oriented API

```python
from intensity_normalization import FCMNormalizer, ZScoreNormalizer
import numpy as np

# Create synthetic brain data
brain_data = np.random.normal(1000, 200, (64, 64, 32))  # WM ~1000
brain_data[20:40, 20:40, 10:20] = np.random.normal(600, 100, (20, 20, 10))  # GM ~600

# FCM normalization (tissue-based)
fcm = FCMNormalizer(tissue_type="wm")  # white matter reference
normalized = fcm.fit_transform(brain_data)

# Z-score normalization
zscore = ZScoreNormalizer()
standardized = zscore.fit_transform(brain_data)

print(f"Original mean: {brain_data[brain_data > 0].mean():.1f}")
print(f"FCM normalized WM mean: {normalized[40:60, 40:60, 10:20].mean():.2f}")
print(f"Z-score mean: {standardized[brain_data > 0].mean():.2f}")
```

### Population-Based Methods

```python
from intensity_normalization import NyulNormalizer, LSQNormalizer
from intensity_normalization.adapters import create_image

# Load multiple subjects
image_paths = ["subject1_t1.nii.gz", "subject2_t1.nii.gz", "subject3_t1.nii.gz"]
images = [create_image(path) for path in image_paths]

# Nyúl histogram matching
nyul = NyulNormalizer(output_min_value=0, output_max_value=100)
nyul.fit_population(images)

# Normalize all images to the same scale
normalized_images = [nyul.transform(img) for img in images]

# LSQ tissue mean harmonization
lsq = LSQNormalizer()
lsq.fit_population(images)
harmonized_images = [lsq.transform(img) for img in images]
```

## Command Line Interface

### Single Image Normalization

```bash
# Basic usage
intensity-normalize fcm brain_t1.nii.gz

# With brain mask
intensity-normalize fcm brain_t1.nii.gz -m brain_mask.nii.gz

# Specify output location
intensity-normalize zscore brain_t1.nii.gz -o normalized_brain.nii.gz

# Different modalities and tissue types
intensity-normalize kde brain_t2.nii.gz --modality t2 --tissue-type gm
intensity-normalize whitestripe brain_flair.nii.gz --modality flair --width 0.1
```

### Method-Specific Parameters

```bash
# FCM with different tissue types and clusters
intensity-normalize fcm brain.nii.gz --tissue-type wm --n-clusters 3

# WhiteStripe with custom width
intensity-normalize whitestripe brain.nii.gz --width 0.05

# Get help for specific methods
intensity-normalize fcm --help
```

## Supported File Formats

Works with all neuroimaging formats supported by nibabel:

| Format | Extensions | Description |
|--------|------------|-------------|
| **NIfTI** | `.nii`, `.nii.gz` | Most common neuroimaging format |
| **FreeSurfer** | `.mgz`, `.mgh` | FreeSurfer volume format |
| **ANALYZE** | `.hdr/.img` | Legacy format pair |
| **MINC** | `.mnc` | Medical Imaging NetCDF |
| **PAR/REC** | `.par/.rec` | Philips scanner format |
| **Numpy** | `.npy` | Raw numpy arrays |

## Normalization Methods

### Individual Methods (Single Image)

| Method | Best For | Description |
|--------|----------|-------------|
| **FCM** | T1-weighted | Fuzzy C-means tissue segmentation (recommended) |
| **Z-Score** | Any modality | Standard score normalization |
| **KDE** | T1/T2/FLAIR | Kernel density estimation of tissue modes |
| **WhiteStripe** | T1-weighted | Normal-appearing white matter standardization |

### Population Methods (Multiple Images)

| Method | Best For | Description |
|--------|----------|-------------|
| **Nyúl** | Cross-scanner | Piecewise linear histogram matching |
| **LSQ** | Multi-site studies | Least squares tissue mean harmonization |

### Method Selection Guide

```python
# T1-weighted images (structural)
normalize_image(t1_image, method="fcm", tissue_type="wm")

# T2-weighted or FLAIR
normalize_image(t2_image, method="kde", modality="t2")

# Quick standardization
normalize_image(image, method="zscore")

# Multi-site harmonization (requires multiple subjects)
from intensity_normalization.services.normalization import NormalizationService
config = NormalizationConfig(method="nyul")
harmonized = NormalizationService.normalize_images(all_images, config)
```

## Architecture Overview

The package is structured as follows:

```
intensity_normalization/
├── domain/          # Core logic
│   ├── protocols.py # Image and normalizer interfaces
│   ├── models.py    # Configuration and value objects
│   └── exceptions.py# Domain-specific exceptions
├── adapters/        # External interfaces
│   ├── images.py    # Universal image adapter (numpy/nibabel)
│   └── io.py        # File I/O operations
├── normalizers/     # Normalization implementations
│   ├── individual/  # Single-image methods (FCM, Z-score, etc.)
│   └── population/  # Multi-image methods (Nyúl, LSQ)
├── services/        # Application services
│   ├── normalization.py # Orchestration logic
│   └── validation.py    # Input validation
└── cli.py          # Command-line interface
```

## Advanced Usage

### Custom Normalizers

```python
from intensity_normalization.domain.protocols import BaseNormalizer
from intensity_normalization.domain.protocols import ImageProtocol

class CustomNormalizer(BaseNormalizer):
    def fit(self, image: ImageProtocol, mask=None) -> 'CustomNormalizer':
        # Implement fitting logic
        self.is_fitted = True
        return self

    def transform(self, image: ImageProtocol, mask=None) -> ImageProtocol:
        # Implement normalization logic
        data = image.get_data()
        normalized_data = your_normalization_function(data)
        return image.with_data(normalized_data)
```

### Configuration-Based Workflow

```python
from intensity_normalization.domain.models import NormalizationConfig, Modality, TissueType
from intensity_normalization.services.normalization import NormalizationService

# Create configuration
config = NormalizationConfig(
    method="fcm",
    modality=Modality.T1,
    tissue_type=TissueType.WM
)

# Validate configuration
from intensity_normalization.services import ValidationService
ValidationService.validate_normalization_config(config)

# Apply normalization
result = NormalizationService.normalize_image(image, config, mask)
```

### Batch Processing

```python
from pathlib import Path
from intensity_normalization.adapters.images import create_image, save_image
from intensity_normalization.services.normalization import NormalizationService

def process_directory(input_dir: Path, output_dir: Path, method: str = "fcm"):
    """Process all NIfTI files in a directory."""
    output_dir.mkdir(exist_ok=True)

    for img_file in input_dir.glob("*.nii.gz"):
        # Load image
        image = create_image(img_file)

        # Create configuration
        config = NormalizationConfig(method=method)

        # Normalize
        normalized = NormalizationService.normalize_image(image, config)

        # Save result
        output_file = output_dir / f"{img_file.stem}_normalized.nii.gz"
        save_image(normalized, output_file)
        print(f"Processed: {img_file.name}")

# Usage
process_directory(Path("raw_images/"), Path("normalized/"))
```

## Development

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/jcreinhold/intensity-normalization.git
cd intensity-normalization

# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --dev
```

### Code Quality

```bash
# Format code
uv run ruff format intensity_normalization/

# Lint code
uv run ruff check intensity_normalization/

# Type checking
uv run mypy intensity_normalization/

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=intensity_normalization --cov-report=html
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Ensure code quality (`uv run ruff format && uv run ruff check --fix && uv run mypy`)
5. Run tests (`uv run pytest`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{reinhold2019evaluating,
  title={Evaluating the impact of intensity normalization on {MR} image synthesis},
  author={Reinhold, Jacob C and Dewey, Blake E and Carass, Aaron and Prince, Jerry L},
  booktitle={Medical Imaging 2019: Image Processing},
  volume={10949},
  pages={109493H},
  year={2019},
  organization={International Society for Optics and Photonics}}
```

## Related Papers

- **FCM**: Udupa, J.K., et al. "A framework for evaluating image segmentation algorithms." Computerized medical imaging and graphics 30.2 (2006): 75-87.
- **Nyúl**: Nyúl, L.G., Udupa, J.K. "On standardizing the MR image intensity scale." Magnetic Resonance in Medicine 42.6 (1999): 1072-1081.
- **WhiteStripe**: Shinohara, R.T., et al. "Statistical normalization techniques for magnetic resonance imaging." NeuroImage 132 (2016): 174-184.

## Support

- **Issues**: [GitHub Issues](https://github.com/jcreinhold/intensity-normalization/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jcreinhold/intensity-normalization/discussions)
