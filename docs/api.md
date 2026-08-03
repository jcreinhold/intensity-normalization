# API reference

```python
import intensity_normalization as inorm
```

numpy arrays in → numpy out; nibabel images in → nibabel images out
(affine/header preserved). float64 inputs stay float64; other dtypes become
float32. All stochastic steps take `seed=` (default 0, deterministic).

## Individual methods

Plain functions of one image — parameters are estimated from the image itself,
so there is nothing to fit or save.

::: intensity_normalization.zscore

::: intensity_normalization.fcm

::: intensity_normalization.kde

::: intensity_normalization.whitestripe

## Population methods

`fit(images, masks)` returns a fitted, callable, savable transform.
`fit_transform` returns the transform and the normalized inputs.

::: intensity_normalization.methods.nyul

::: intensity_normalization.methods.lsq

::: intensity_normalization.methods.ravel

## Array-level methods

Every wrapper above delegates to a pure-numpy core named after it plus
`_array`. Cores take a required boolean foreground mask and resolved options —
no nibabel, no mask estimation, no modality strings. Use them when your
pipeline already holds arrays.

::: intensity_normalization.methods.zscore.zscore_array

::: intensity_normalization.methods.fcm.fcm_array

::: intensity_normalization.methods.kde.kde_array

::: intensity_normalization.methods.whitestripe.whitestripe_array

::: intensity_normalization.methods.nyul.fit_array

::: intensity_normalization.methods.lsq.fit_array

::: intensity_normalization.methods.ravel.fit_transform_array

## Tools

::: intensity_normalization.tissue_membership

::: intensity_normalization.plot_histograms

::: intensity_normalization.preprocess

::: intensity_normalization.coregister

## Histogram utilities

::: intensity_normalization.histogram

## Types

Signatures across the package use one vocabulary, importable from the root:

| Name | Meaning |
|---|---|
| `Image` | an intensity array or a nibabel spatial image |
| `Mask` | a float or bool array, or a nibabel image |
| `IntensityArray` | float image data (numpy) |
| `MaskArray` | a mask as an array (float or bool) |
| `BinaryMask` | a thresholded boolean mask |
| `ForegroundIntensities` | 1-D samples inside a foreground mask |

## Errors

::: intensity_normalization.errors
