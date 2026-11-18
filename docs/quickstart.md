# Quickstart

A five-minute lesson: install, normalize one T1-w image, check the result.

## Install

```bash
pip install intensity-normalization  # or: uv add intensity-normalization
```
Alternatively, you can install via conda:

```sh
conda install conda-forge::intensity-normalization
```

## Get a brain mask

Most methods expect a brain mask — a rough one is fine, it only needs to remove most non-brain tissue.
[ROBEX](https://www.nitrc.org/projects/robex), HD-BET, and SynthStrip all work. If the image is already skull-stripped,
skip the mask: the foreground is estimated as the positive voxels.

## Normalize one image

FCM normalization is a good default for T1-w brain images: it scales the white-matter mean to 1.

=== "CLI"

    ```bash
    intensity-normalize fcm t1w.nii.gz -m brain_mask.nii.gz -o t1w_norm.nii.gz
    ```

=== "Python"

    ```python
    import nibabel as nib
    import intensity_normalization as inorm

    image = nib.load("t1w.nii.gz")
    mask = nib.load("brain_mask.nii.gz")

    normed = inorm.fcm(image, mask=mask)
    nib.save(normed, "t1w_norm.nii.gz")
    ```

The output has the same type as the input: numpy in → numpy out, nibabel in → nibabel out, affine and header preserved.

## Check the result

Plot the foreground histogram before and after — the fastest way to catch a bad mask. Install the plotting extra and
rerun with `-p`:

```bash
pip install "intensity-normalization[plot]"  # or: uv add ...
intensity-normalize fcm t1w.nii.gz -m brain_mask.nii.gz -p
```

After FCM normalization of T1-w images, the white-matter peaks should sit at the norm value (1 by default).

## Where next

- Normalize a dataset, validate a batch, save a fitted transform: [How-to guides](how-to.md)
- Pick a method for your data: [Choosing a method](methods.md)
- The math behind each method: [How the methods work](algorithms.md)
