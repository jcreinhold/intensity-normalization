=============
Example usage
=============

Individual timepoint-based normalization
========================================

The below section focuses on Fuzzy C-means (FCM)-based normalization, but can be used as a reference for all individual
timepoint-based normalization methods.

Once the package is installed, if you just want to do some sort of normalization and not think too much about it, a
reasonable choice is Fuzzy C-means (FCM)-based normalization. Note that FCM requires access to a
(*non-gadolinium-enhanced*) T1-w image, if this is not possible then I would recommend doing either z-score or KDE
normalization for simple normalization tasks. The FCM method also requires a brain mask for the image, although the
brain mask need not be perfect
(`ROBEX <https://sites.google.com/site/jeiglesias/ROBEX>`_ works fine for this purpose).

Note that FCM-based normalization acts on the image by calculating the specified tissue mean, e.g., white matter (WM)
mean and setting that to a specified value (the default is 1 in the code base although that is a tunable parameter).
Our FCM-based normalization method requires that a timepoint contains a T1-w image. We use the T1-w image and the brain
mask to create a tissue mask over which we calculate the tissue mean. We then normalize as previously stated (see
`here <https://intensity-normalization.readthedocs.io/en/latest/algorithm.html#fuzzy-c-means>`_ for more detail).
This tissue mask can then be used to normalize the remaining contrasts in the set of images for a specific patient
assuming that the remaining contrast images are registered to the T1-w image.

Since most of the command line interfaces (CLIs) are installed along with the package, we can run ``fcm-normalize``
in the terminal to normalize a T1-w image and create a WM mask by running the following command (replacing paths as
necessary)::

    fcm-normalize t1w_image.nii.gz -m brain_mask.nii.gz -o t1w_norm.nii.gz -v -mo t1 -tt wm

This will output the normalized T1-w image to `t1w_norm.nii.gz` and will additionally create a tissue
mask for the WM in the same directory with `wm_membership` appended to the filename. You can then input
the WM membership back in to the program to normalize an image of a different contrast, e.g. for T2::

    fcm-normalize t2w_image.nii.gz -tm t1w_image_wm_membership.nii.gz -o t2w_norm.nii.gz -v -mo t2

You can run ``fcm-normalize -h`` to see more options, but the above covers most of the details necessary to
run FCM normalization on a single image.

You can process a directory of NIfTI images like this::

    find -L t1w_image_dir -type f -name '*.nii*' -exec fcm-normalize "{}" \;

and it will FCM normalize all the images in the directory ``t1w_image_dir/`` assuming all the images are
skull-stripped.

If you want to quickly inspect the normalization results on a directory (as in the last command), you can append the
``-p`` flag which will create a plot of the histograms inside the brain mask of the normalized image (or images if you're
using a sample-based method).

For the above case, you should expect to see alignment around the intensity level of 1 (or whatever the ``--norm-value``
is set to). You can also use the ``plot-histograms`` CLI which is also installed (`see
here <https://intensity-normalization.readthedocs.io/en/latest/exec.html#plotting>`_ for documentation). A use case of
the ``plot-histograms`` command would be to plot the histograms of all individual timepoint-based normalized images,
or to inspect the histograms of a set of images before and after normalization.

Example usage on a directory for sample-based methods
=====================================================

The sample-based normalization CLIs (RAVEL, Nyul, and LSQ) operate on a directory of NIfTI images (either 2D or 3D).
That is, suppose you have a directory of images ``img_dir`` that contains NIfTI (.nii.gz or .nii) images, like so::

    ├── img_dir
    │   ├── img1.nii.gz
    │   ├── img2.nii.gz
    │   ├── img3.nii.gz
    │   ├── ...
    │   ├── imgN.nii.gz

In addition to the images, the normalization CLIs also can take brain masks as input; the masks (or absence of masks)
can affect normalization quality. If you have brain masks for the corresponding images (for example, the images in
``img_dir``), they should be setup like so::

    ├── mask_dir
    │   ├── mask1.nii.gz
    │   ├── mask2.nii.gz
    │   ├── mask3.nii.gz
    │   ├── ...
    │   ├── maskN.nii.gz

Note that when both ``img_dir`` and ``mask_dir`` are sorted alphabetically, each mask should correspond to the correct
image. Other than that, the name of the image or mask is not important.

If you have a setup as shown above (with ``img_dir`` and ``mask_dir``), you can call any sample-based
normalization CLI on ``img_dir`` to normalize all images in that directory. For example,
with ``nyul-normalize`` (assuming that ``img_dir`` contains T1-w images, in this example) the
call would be something like::

    nyul-normalize img_dir -m mask_dir -o out_dir -v -mo t1 -tt wm

Additional Provided Routines
============================

There a variety of other routines provided for analysis and preprocessing. The CLI names are:

1) ``coregister`` - coregister via a rigid and affine transformation
2) ``plot-histograms`` - plot the histograms of a directory of images on one figure for comparison
3) ``tissue-mask`` - create a tissue mask of an input image
4) ``preprocess`` - resample, N4-correct, and reorient the image and mask

Importing normalization methods
===============================

While in this tutorial we discussed interfacing with the package through command line interfaces (CLIs),
it is worth noting that the normalization routines (and other utilities) are available as importable python functions
which you can import, e.g.

.. code-block:: python

   import nibabel as nib
   from intensity_normalization.normalize.fcm import FCMNormalizer

   image = nib.load("test_t1w_image.nii")  # assume skull-stripped otherwise load mask too

   fcm_norm = FCMNormalizer(tissue_type="wm")
   normalized = fcm_norm(image) # alternatively, you can pass in a numpy array which will return a numpy array
   normalized.to_filename("normalized_test_t1w_image.nii")  # this works if you passed in a nibabel Nifti image
   # or if you want to do further processing on the data array
   norm_data = normalized.get_fdata()  # if you passed in a nibabel Nifti image, otherwise normalized is an array

   # now normalize the co-registered, corresponding T2-w image
   t2w_image = nib.load("test_t2w_image.nii")
   t2w_normalized = fcm_norm(t2w_image, modality="t2")

Generally, the normalization methods have a similar interface, although some methods (RAVEL, Nyul, and LSQ) require a
list of images (and, optionally, corresponding masks), like so:

.. code-block:: python

   normalizer = NormalizerClass(**init_args)
   normalizer(image, mask, modality)

where ``init_args`` is a dictionary of method dependent keyword arguments, ``image`` is either a nibabel NIfTI image or
a numpy array; ``mask`` is one of ``None`` (or not provided), a nibabel NIfTI image, or a numpy array; ``modality`` is a
string representing the modality.
