# Migrating to v4

v4 is a clean break from v2/v3: new package layout, new Python API, one CLI. The algorithms are the same (and better
tested); the interface was redesigned.

## Since v4.0: boundary-resolved options

`ravel.fit_transform` takes `whitestripe=WhiteStripeSpec(...)` instead of `whitestripe_kwargs=dict(...)` — typed,
documented in one place, typo-proof:

```python
# before
ravel.fit_transform(images, whitestripe_kwargs={"width": 0.1})
# after
from intensity_normalization.methods.whitestripe import WhiteStripeSpec

ravel.fit_transform(images, whitestripe=WhiteStripeSpec(width=0.1))
```

`histogram.tissue_mode` no longer accepts `modality=`; it takes a required `peak=`. Resolve modality names first with
the new `histogram.resolve_peak`:

```python
# before
histogram.tissue_mode(intensities, modality="t1")
# after
histogram.tissue_mode(intensities, peak=histogram.resolve_peak("t1", None))
```

## CLI

Eleven separate scripts became subcommands of one binary:

| v2 script | v4 command |
| --- | --- |
| `zscore-normalize img.nii -m mask.nii` | `intensity-normalize zscore img.nii.gz -m mask.nii.gz` |
| `fcm-normalize` | `intensity-normalize fcm` |
| `kde-normalize` | `intensity-normalize kde` |
| `ws-normalize` | `intensity-normalize whitestripe` |
| `nyul-normalize dir/ -m masks/ -ssh s.npy` | `intensity-normalize nyul dir/ -m masks/ --save-state s.npz` |
| `nyul-normalize dir/ -lsh s.npy` | `intensity-normalize nyul dir/ --load-state s.npz` |
| `lsq-normalize` | `intensity-normalize lsq` |
| `ravel-normalize` | `intensity-normalize ravel` |
| `tissue-membership` | `intensity-normalize tissue-membership` |
| `plot-histograms` | `intensity-normalize plot-histograms` |
| `preprocess` | `intensity-normalize preprocess` |
| `coregister` | `intensity-normalize coregister` |

Single-image and directory CLIs were unified: individual methods accept one image, several images, or a directory. Saved
transform state moved from `.npy` to stamped `.npz` (old `.npy` files are not loadable — refit them).

## Python API

Classes with `setup`/`teardown` and a `modality` keyword became:

- **individual methods**: plain functions — `inorm.whitestripe(img, mask=mask)`
- **population methods**: `fit` returns a fitted transform — `tx = inorm.nyul.fit(imgs); tx(new_img); tx.save("x.npz")`

```python
# v2
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize

ws = WhiteStripeNormalize(norm_value=1.0)
normalized = ws(image, mask)

# v4
import intensity_normalization as inorm

normalized = inorm.whitestripe(image, mask=mask)
```

Notable changes:

- `pymedio` images are gone; pass numpy arrays or nibabel images and get the same type back.
- `scikit-fuzzy` is gone; fuzzy c-means is implemented in-house (seeded, deterministic by default).
- `Modality`/`TissueType` enums became plain validated strings (`"t1"`, `"wm"`).
- The `normalize_image(method="...")` dispatcher (v3) was removed; call the method by name.
- antspy/matplotlib are optional extras as before, now imported lazily: `[ants]`, `[plot]`.
