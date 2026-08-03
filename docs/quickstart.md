# Quickstart

Five minutes to normalized images. Install with `pip install intensity-normalization`.

## A note on masks

Most methods expect a **brain mask** (or an already skull-stripped image). The
mask doesn't need to be perfect — it just needs to remove most non-brain tissue.
[ROBEX](https://www.nitrc.org/projects/robex), HD-BET, and SynthStrip all work.
If the image is already skull-stripped, pass no mask: the foreground is
estimated as the positive voxels.

## Individual methods (per-image)

If you just want one reasonable choice: **FCM-based normalization on a T1-w
image** (non-gadolinium-enhanced). No T1-w? Use `zscore` or `kde`.

=== "CLI"

    ```bash
    intensity-normalize fcm t1w.nii.gz -m brain_mask.nii.gz -o t1w_norm.nii.gz
    ```

    Batch a whole directory (masks matched by filename), in parallel:

    ```bash
    intensity-normalize fcm images/ --mask-dir masks/ --output-dir normalized/ -j 8
    ```

=== "Python"

    ```python
    import intensity_normalization as inorm

    normed = inorm.fcm(t1w_image, mask=brain_mask)          # wm mean -> 1
    normed = inorm.fcm(t1w_image, mask=brain_mask, tissue="gm", norm_value=2.0)
    ```

    `normed` is the same type as the input: numpy in → numpy out, nibabel in →
    nibabel out (affine and header preserved).

### Other contrasts of the same subject

FCM tissue memberships come from the T1-w image. To normalize a co-registered
T2-w (or other contrast) of the same subject with the same tissue statistics:

```python
membership = inorm.tissue_membership(t1w_image, mask=brain_mask)  # (..., 3): CSF/GM/WM
t2_normed = inorm.fcm(t2w_image, mask=t2_mask, modality="t2", membership=membership[..., 2])
```

## Validate visually (do this!)

Whatever method you pick, **plot the foreground histograms before and after** —
it's the fastest way to catch a bad mask or a wrong modality choice:

```bash
intensity-normalize plot-histograms normalized/ -m masks/ -o after.png
intensity-normalize whitestripe t1w.nii.gz -m mask.nii.gz -p   # before/after inline
```

After FCM normalization of T1-w images you should see the white-matter peaks
aligned at the norm value (default 1).

## Population methods (fit on a set, apply to new scans)

`nyul`, `lsq`, and `ravel` learn a transform from a set of images. The learned
transform can be saved and applied to new data later — fit on your training set
once, apply at inference time.

=== "CLI"

    ```bash
    # fit on a directory, normalize it, and save the learned transform
    intensity-normalize nyul train_images/ -m train_masks/ -o normalized/ --save-state nyul.npz

    # later: apply the saved transform to new images
    intensity-normalize nyul new_images/ -m new_masks/ -o normalized_new/ --load-state nyul.npz
    ```

=== "Python"

    ```python
    import intensity_normalization as inorm

    tx = inorm.nyul.fit(train_images, masks=train_masks)
    tx.save("nyul.npz")

    normed = tx(new_image)                       # callable, type-preserving
    tx = inorm.NyulTransform.load("nyul.npz")    # reload later
    ```

!!! note "RAVEL is different"
    RAVEL is a *batch* correction: it removes across-image technical variation,
    so the images being corrected must be part of the fit. There is no
    apply-to-one-new-scan transform. It also requires co-registered,
    same-shape images — see `intensity-normalize coregister`.

## Tools

| Command | What it does | Extra |
|---|---|---|
| `tissue-membership` | CSF/GM/WM fuzzy membership maps of a T1-w image | — |
| `plot-histograms` | foreground histograms of a directory, one figure | `[plot]` |
| `preprocess` | N4 bias correction + resample + reorient | `[ants]` |
| `coregister` | register images to a template (MNI by default) | `[ants]` |

```bash
intensity-normalize preprocess t1w.nii.gz -m mask.nii.gz -r 1 1 1
intensity-normalize coregister img1.nii.gz img2.nii.gz --template mni.nii.gz -o registered/
```
