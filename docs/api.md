# API reference

```python
import intensity_normalization as inorm
```

numpy arrays in → numpy out; nibabel images in → nibabel images out
(affine/header preserved). All stochastic steps take `seed=` (default 0,
deterministic).

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

## Tools

::: intensity_normalization.tissue_membership

::: intensity_normalization.plot_histograms

::: intensity_normalization.preprocess

::: intensity_normalization.coregister

## Histogram utilities

::: intensity_normalization.histogram

## Errors

::: intensity_normalization.errors
