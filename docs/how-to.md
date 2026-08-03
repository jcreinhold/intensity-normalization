# How-to guides

Task-oriented recipes. Each assumes the package is installed; plotting recipes also need the `[plot]` extra, ANTs
recipes need `[ants]`, e.g. `pip install "intensity-normalization[plot]"` (or `uv add`).

## Normalize a directory of images

Point the CLI at a directory. Masks are matched by filename from `--mask-dir`, and `-j` runs the batch in parallel:

```bash
intensity-normalize fcm images/ --mask-dir masks/ --output-dir normalized/ -j 8
```

Outputs are named `<input>_<method>.nii.gz` inside the output directory.

From Python, loop over the files — or use a population method, which takes lists directly (see below).

## Normalize other contrasts of the same subject

FCM tissue memberships come from the T1-w image. Reuse them for a co-registered T2-w (or other contrast) of the same
subject:

```python
membership = inorm.tissue_membership(t1w_image, mask=brain_mask)  # (..., 3): CSF/GM/WM
t2_normed = inorm.fcm(t2w_image, mask=t2_mask, modality="t2", membership=membership[..., 2])
```

## Fit once, apply to new scans

`nyul` and `lsq` learn a transform you can save and reapply — fit on the training set, apply the *same* transform to
validation, test, and inference data:

=== "CLI"

    ```bash
    intensity-normalize nyul train/ -m train_masks/ -o train_norm/ --save-state nyul.npz
    intensity-normalize nyul new/ -m new_masks/ -o new_norm/ --load-state nyul.npz
    ```

=== "Python"

    ```python
    tx = inorm.nyul.fit(train_images, masks=train_masks)
    tx.save("nyul.npz")

    normed = tx(new_image)
    tx = inorm.NyulTransform.load("nyul.npz")  # reload later
    ```

RAVEL has no apply-later mode: it corrects the batch it was fit on.

## Validate results with histogram plots

Plot the foreground histograms of a directory before and after normalizing:

```bash
intensity-normalize plot-histograms images/ -m masks/ -o before.png
intensity-normalize plot-histograms normalized/ -m masks/ -o after.png
```

Or see before/after inline while normalizing one image with `-p`. Peaks for the same tissue should line up across images
afterward — white matter at the norm value for FCM on T1-w. If they don't, suspect the masks first: bad or misaligned
masks cause most bad results.

## Work at the array level

Every method has an array core named after its wrapper plus `_array` (`zscore_array`, `fcm_array`, `whitestripe_array`,
`kde_array`, `tissue_membership_array`, `fit_array`, `transform_array`, `fit_transform_array`). Cores are pure numpy: no
nibabel, no mask estimation, no modality strings. Use them inside your own pipeline when you already hold arrays:

```python
from intensity_normalization.methods.zscore import zscore_array

foreground = data > 0  # a boolean mask, required
normed = zscore_array(data, foreground)
```

The wrappers (`inorm.zscore`, …) add nibabel handling, mask estimation, and modality-name resolution on top of these
cores. The type vocabulary used in their signatures — `Image`, `Mask`, `IntensityArray`, `MaskArray`, `BinaryMask`,
`ForegroundIntensities` — is importable from the package root for annotating your own code.

## Remove batch effects with RAVEL

RAVEL removes across-image technical variation, so its inputs must be voxel-aligned: same shape, deformably
co-registered. Either let it register internally (default, needs `[ants]`) or pre-register and pass `--no-registration`:

```bash
intensity-normalize coregister imgs/*.nii.gz --template mni.nii.gz -o registered/
intensity-normalize ravel registered/ -m masks/ -o corrected/ --no-registration
```

## Preprocess before normalizing

N4 bias-field correction, optional resampling and reorientation:

```bash
intensity-normalize preprocess t1w.nii.gz -m mask.nii.gz -r 1 1 1  # [ants]
```
