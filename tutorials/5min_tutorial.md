# Quick Tutorial

## Fuzzy C-means-based Normalization

Once the package is installed, if you just want to do some sort of normalization and not think too much about it, a reasonable choice is Fuzzy C-means (FCM)-based
normalization. Note that FCM requires access to a (_non-gadolinium-enhanced_) T1-w image, if this is not possible then I would recommend doing either z-score or KDE normalization
for simple normalization tasks. The FCM method also requires a brain mask for the image, although the brain mask need not be perfect
([ROBEX](https://sites.google.com/site/jeiglesias/ROBEX) works fine for this purpose).

Note that FCM-based normalization acts on the image by calculating the specified tissue mean, e.g., white matter (WM) mean
and setting that to a specified value (the default is 1 in the code base although that is a tunable parameter). Our FCM-based normalization method requires that
a set of scans contain a T1-w image. We use the T1-w image and the brain mask to create a tissue mask over which we calculate the tissue mean.
We then normalize as previously stated (see [here](https://intensity-normalization.readthedocs.io/en/latest/algorithm.html#fuzzy-c-means) for more detail).
This tissue mask can then be used to normalize the remaining contrasts in the set of images for a specific patient assuming that the
remaining contrast images are registered to the T1-w image.

Since all the command line interfaces (CLIs) are installed along with the package, we can run `fcm-normalize`
in the terminal to normalize a T1-w image and create a WM mask by running the following command (replacing paths as necessary):

```bash
fcm-normalize -i t1_w_image_path.nii.gz -m brain_mask_path.nii.gz -o t1_norm_path.nii.gz -v -c t1 -s -tt wm
```

This will output the normalized T1-w image to `t1_norm_path.nii.gz` and will create a directory
called `tissue_masks` in which the WM mask will be saved. You can then input the WM mask back in to
the program to normalize an image of a different contrast, e.g. for T2,

```bash
fcm-normalize -i t2_image_path.nii.gz -tm wm_masks/wm_mask.nii.gz -o t2_norm_path.nii.gz -v -c t2
```

You can run `fcm-normalize -h` to see more options, but the above covers most of the details necessary to
run FCM normalization on a single image.  You can also input a directory of images like this:

```bash
fcm-normalize -i t1_imgs/ -m brain_masks/ -o out/ -v -c t1
```

and it will FCM normalize all the images in the directory `t1_imgs/` so long as the number of images and brain masks
are equal and correspond to one another and output the normalized images into the directory `out/`.

If you want to quickly inspect the normalization results on a directory (as in the last command), you can append the
`-p` flag which will create a plot of the histograms inside the brain mask of the normalized images. For the above
case, you should expect to see alignment around the intensity level of 1 (or whatever the `--norm-value` is set to).
You can also use the `plot-hists` CLI which is also installed (see [here](https://intensity-normalization.readthedocs.io/en/latest/exec.html#plotting)
for documentation). A use case of the `plot-hists` command would be to inspect the histograms of a set of images *before* normalization
to compare with the results of normalization.

## Example usage on a directory

All CLIs for normalization (and preprocessing) can operate on a directory of NIfTI images (either 2D or 3D). That is,
suppose you have a directory of images `img_dir` that contains NIfTI (.nii.gz or .nii) images, like so:

```none
├── img_dir
│   ├── img1.nii.gz
│   ├── img2.nii.gz
│   ├── img3.nii.gz
│   ├── ...
│   ├── imgN.nii.gz
```

In addition to the images, all normalization CLIs also can take brain masks as input; the masks (or absence of masks) can affect normalization quality.
If you have brain masks for the corresponding images (for example, the images in `img_dir`), they should be setup
like so:

```none
├── mask_dir
│   ├── mask1.nii.gz
│   ├── mask2.nii.gz
│   ├── mask3.nii.gz
│   ├── ...
│   ├── maskN.nii.gz
```

Note that when both `img_dir` and `mask_dir` are sorted alphabetically, each mask should correspond to the correct image.
Other than that, the name of the image or mask is not important.

If you have a setup as shown above (with `img_dir` and `mask_dir`), you can call any
normalization CLI on `img_dir` to normalize all images in that directory. For example,
with `fcm-normalize` (assuming that `img_dir` contains T1-w images, in this example) the
call would be something like:

```bash
fcm-normalize -i img_dir -m mask_dir -o out_dir -v -c t1 -tt wm
```

## Other Normalization Methods

The other methods not listed above are accessible via:

1) `zscore-normalize` - do z-score normalization over the brain mask
2) `ws-normalize` - WhiteStripe normalization
3) `ravel-normalize` - RAVEL normalization
4) `kde-normalize` - Kernel Density Estimate WM peak normalization
5) `nyul-normalize` - Nyul & Udupa Piecewise linear affine normalization based on learned histogram features
6) `gmm-normalize` - use a GMM to normalize the WM mean over the brain mask, like FCM (do not recommend using this method!)
7) `lsq-normalize` - minimize the least-squares distance between the means of CSF, GM, and WM in a set of images

Note that these all have approximately the same interface with the `-i`, `-m` and `-o` options, but each
individual method *may* need some additional input. To determine if this is the case you can either view the
[executable documentation here](https://intensity-normalization.readthedocs.io/en/latest/exec.html) or run the command line interface (CLI) with the `-h`
or `--help` option. To get more detail about what each of these algorithms actually does
see the [algorithm documentation here](https://intensity-normalization.readthedocs.io/en/latest/algorithm.html).

## Additional Provided Routines

There a variety of other routines provided for analysis and preprocessing. The CLI names are:

1) `coregister` - coregister via a rigid and affine transformation
2) `plot-hists` - plot the histograms of a directory of images on one figure for comparison
3) `tissue-mask` - create a tissue mask of an input image
4) `preprocess` - resample, N4-correct, and reorient the image and mask

## Final Note

While in this tutorial we discussed interfacing with the package through command line interfaces (CLIs),
it is worth noting that the normalization routines (and other utilities) are available as importable python functions
which you can import, e.g.,

```python
from intensity_normalization.normalize import fcm
wm_mask = fcm.find_tissue_mask(img, brain_mask, tissue_type="wm")
normalized = fcm.fcm_normalize(img, wm_mask)
```
