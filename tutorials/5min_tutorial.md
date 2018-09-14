# Quick Tutorial

## Install package

First download the package through git, i.e.,

`git clone https://github.com/jcreinhold/intensity-normalization.git`

If you are using conda, the easiest way to ensure you are up and running is to run the  `create_env.sh` script
located in the main directory. You run that as follows:

`. ./create_env.sh`

If you are *not* using conda, then you can try to install the package via the setup.py script, i.e.,
inside the `intensity-normalization` directory, run the following command:

`python setup.py install`

If you don't want to bother with any of this, you can create a Docker image or Singularity image via:

`docker pull jcreinhold/intensity-normalization`

or 

`singularity pull docker://jcreinhold/intensity-normalization`


## Fuzzy C-means-based Normalization

Once the package is installed, if you just want to do some sort of normalization and not think about it, a reasonable choice is Fuzzy C-means (FCM)-based
normalization and Gaussian Mixture Model (GMM)-based normalization; these are covered in this section and the next section respectively.

We can do Fuzzy C-means-based white matter (WM) mean normalization as follows:

```bash
fcm-normalize -i t1_w_image_path.nii.gz -m mask_path.nii.gz -o t1_norm_path.nii.gz -v
```
 
This will output the normalized T1-w image to `output_path.nii.gz` and will create a directory 
called `wm_masks` in which the WM mask will be saved. You can then input the WM mask back in to 
the program to normalize an image of a different contrast, e.g. for T2,

```bash
fcm-normalize -i t2_image_path.nii.gz -w wm_masks/wm_mask.nii.gz -o t2_norm_path.nii.gz -v -c t2
``` 
 
You can run `fcm-normalize -h` to see more options, but the above covers most of the details necessary to 
run FCM normalization on a single image.  You can also input a directory of images like this:

```bash
fcm-normalize -i t1_imgs/ -m brain_masks/ -o out/ -v
``` 
 
and it will FCM normalize all the images in the directory so long as the number of images and brain masks 
are equal and correspond to one another.

## Gaussian Mixture Model-based Normalization 

A GMM-based normalization method is automatically 
installed and accessible via the command-line to your path like `fcm-normalize` and is called 
`gmm-normalize`. That one is simpler to run because no 
intermediate WM mask is created, but it only supports T1, T2, and FLAIR. So you would just call:

```bash
gmm-normalize -i imgs/ -m masks/ -o norm/ -v
``` 

## Other Normalization Methods

The other methods not listed above are accessible via:

1) `zscore-normalize` - do z-score normalization over the brain mask
2) `ws-normalize` - WhiteStripe normalization
3) `ravel-normalize` - RAVEL normalization
4) `kde-normalize` - Kernel Density Estimate WM peak normalization
5) `hm-normalize` - Nyul & Udupa Piecewise linear histogram matching normalization

Note that these all have approximately the same interface with the `-i`, `-m` and `-o` options, but each 
individual method *may* need some additional input. To determine if this is the case you can either run the 
command line interface (CLI) or run the CLI with the `-h` or `--help` option.

## Additional Provided Routines

There a variety of other routines provided for analysis and preprocessing. The CLI names are:

1) `coregister` - coregister via a rigid and affine transformation 
2) `plot-hists` - plot the histograms of a directory of images on one figure for comparison
3) `tissue-mask` - create a tissue mask of an input image
4) `preprocess` - resample, N4-correct, and reorient the image and mask
