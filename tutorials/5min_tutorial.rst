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
brain mask doesn't need to be perfect
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

This will output the normalized T1-w image to ``t1w_norm.nii.gz`` and will additionally create a tissue
mask for the WM in the same directory with ``wm_membership`` appended to the filename. You can then input
the WM membership back in to the program to normalize an image of a different contrast, e.g. for T2::

    fcm-normalize t2w_image.nii.gz -tm t1w_image_wm_membership.nii.gz -o t2w_norm.nii.gz -v -mo t2

You can run ``fcm-normalize -h`` to see more options, but the above covers most of the details necessary to
run FCM normalization on a single image.

You can process a directory of images like this::

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

The sample-based normalization CLIs (RAVEL, Nyul, and LSQ) operate on a directory of images (either 2D or 3D).
That is, suppose you have a directory of images ``img_dir`` that contains images, like so::

    ├── img_dir
    │   ├── img1.ext
    │   ├── img2.ext
    │   ├── img3.ext
    │   ├── ...
    │   ├── imgN.ext

In addition to the images, the normalization CLIs also can take brain masks as input; the masks (or absence of masks)
can affect normalization quality. If you have brain masks for the corresponding images (for example, the images in
``img_dir``), they should be setup like so::

    ├── mask_dir
    │   ├── mask1.ext
    │   ├── mask2.ext
    │   ├── mask3.ext
    │   ├── ...
    │   ├── maskN.ext

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

1) ``plot-histograms`` - plot the histograms of a directory of images on one figure for comparison
2) ``tissue-membership`` - find and output tissue membership of an input image

The following (along with ``ravel-normalize``) are available only if you install
``intensity-normalization`` with ``pip install "intensity-normalization[ants]"``.

1) ``coregister`` - coregister via a rigid and affine transformation
2) ``preprocess`` - resample, N4-correct, and reorient the image and mask

Python API for normalization methods
====================================

While in this tutorial we discussed interfacing with the package through command line interfaces (CLIs),
it is worth noting that the normalization routines (and other utilities) are available as via a Python API
which you can import into your project or script, e.g.,

.. code-block:: python

   import nibabel as nib
   from intensity_normalization.typing import Modality, TissueType
   from intensity_normalization.normalize.fcm import FCMNormalize

   image = nib.load("test_t1w_image.nii").get_fdata()  # assume skull-stripped otherwise load mask too

   fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
   normalized = fcm_norm(image)

   # now normalize the co-registered, corresponding T2-w image
   t2w_image = nib.load("test_t2w_image.nii").get_fdata()
   t2w_normalized = fcm_norm(t2w_image, modality=Modality.T2)

   # to use a brain mask instead of a skull-stripped image do this:
   mask = nib.load("brain_mask.nii").get_fdata()
   normalized_t1w = fcm_norm(image, mask)
   # the WM mask is an attribute in the class, so normalize the t2 with:
   normalized_t2w = fcm_norm(t2w_image, modality=Modality.T2)

   # make a new instance of the normalizer to normalize a new image, i.e.:
   new_image = nib.load("test_t1w_image_2.nii")
   fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
   normalized = fcm_norm(new_image.get_fdata())

   # you can save the normalized image with nibabel as follows:
   nib.Nifti1Image(normalized, new_image.affine).to_filename("normalized.nii")

Generally, the normalization methods have a similar interface, although some methods (RAVEL, Nyul, and LSQ) require a
list of images (and, optionally, corresponding masks), like so:

.. code-block:: python

   normalizer = NormalizerClass(**init_args)
   normalizer(image, mask, modality)

where ``init_args`` is a dictionary of method dependent keyword arguments, ``image`` is something like a
a numpy array (i.e., the pixel data of an image, see `pymedio <https://github.com/jcreinhold/pymedio>`_ for
a flexible package to open various types of medical image; it returns them as a subclass of np.ndarray);
``mask`` is one of ``None`` (or not provided), or something like a numpy array (like ``image``); ``modality`` is a
string representing the modality.

Opening and normalizing images with ``pymedio``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are many medical image readers available in Python, and so long as you can convert the pixel/voxel data
to a numpy array, you use them. All that ``intensity-normalization`` requires is an array-like data type.

The above shows an example with ``nibabel`` and for NIfTI images. If you have DICOM images or
other formats, a flexible image reader that requires minimal storage and dependencies is
`pymedio <https://github.com/jcreinhold/pymedio>`_. Assuming you install ``pymedio`` like
``pip install "pymedio[all]"``, an example opening an image and normalizing it is shown below.

Assume ``test_t1w_image`` is a directory of DICOM images:

.. code-block:: python

   import pymedio.image as mioi
   from intensity_normalization.typing import Modality, TissueType
   from intensity_normalization.normalize.fcm import FCMNormalize

   image = mioi.Image.from_path("test_t1w_image/")  # assume skull-stripped otherwise load mask too

   fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
   normalized = fcm_norm(image)

   # now normalize the co-registered, corresponding T2-w image
   t2w_image = mioi.Image.from_path("test_t2w_image.nii")  # or some other extension/directory of DICOM
   t2w_normalized = fcm_norm(t2w_image, modality=Modality.T2)

   # to use a brain mask instead of a skull-stripped image do this:
   mask = mioi("brain_mask.nii")
   normalized_t1w = fcm_norm(image, mask)
   # the WM mask is an attribute in the class, so normalize the t2 with:
   normalized_t2w = fcm_norm(t2w_image, modality=Modality.T2)

   # you can save the normalized image with pymedio as follows:
   normalized_t2w.to_filename("normalized.nii")

``pymedio`` images can be used everywhere in ``intensity-normalization`` that a numpy array can be used.
It will hold the affine transformation matrix as an attribute (at ``.affine``) and can be operated on like
a numpy array without losing the affine transformation matrix.

Validating normalization results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should validate the results of normalization by plotting the histograms of the
foreground image intensities before and after normalization, e.g.,

.. code-block:: python

   from intensity_normalization.plot.histogram import HistogramPlotter, plot_histogram

   import matplotlib.pyplot as plt
   import nibabel as nib
   from intensity_normalization.typing import Modality, TissueType
   from intensity_normalization.normalize.fcm import FCMNormalize

   image = nib.load("test_t1w_image.nii").get_fdata()
   mask = nib.load("test_t1w_brain_mask.nii").get_fdata()

   fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
   normalized = fcm_norm(image, mask, modality=Modality.T1)

   plot_histogram(image, mask)
   plt.title("Unnormalized")
   plt.show()

   plot_histogram(norm, mask)
   plt.title("FCM Normalized")
   plt.show()

   # or if you have a set of images
   images = [nib.load(fn).get_fdata() for fn in filenames]
   masks = [nib.load(fn).get_fdata() for fn in mask_filenames]
   normed = [fcm_norm(img, msk) for img, msk in zip(images, masks)]
   hp = HistogramPlotter(title="FCM Normalized")
   _ = hp(images, masks)
   plt.show()

Another Python API example (co-registration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``intensity-normalization`` relies on `ANTsPy <https://github.com/ANTsX/ANTsPy>`_ to do registration, so,
for this example, you'll need to install ANTsPy first. You'll likely need to let it compile from source
(~40 minutes) which requires `CMake <https://cmake.org/>`_ [*]_.

Once you have ANTsPy installed, you can co-register an image like:

.. code-block:: python

   # load the images
   import nibabel as nib
   image = nib.load("path/to/image.nii")
   target = nib.load("path/to/target.nii")

   # setup up registration
   from intensity_normalization.util.coregister import register
   transformation = "Affine"
   interpolator = "bSpline"
   initial_rigid = True  # do initial rigid transformation before transformation

   # verify this is a supported transformation, interpolator
   from intensity_normalization.typing import (
       allowed_transformations, allowed_interpolators
   )
   assert transformation in allowed_transformations
   assert interpolator in allowed_interpolators

   # register the image to the target
   registered = register(
       image,
       target,
       type_of_transform=transformation,
       interpolator=interpolator,
       initial_rigid=initial_rigid
   )

   # save the image or get the registered image out
   registered.to_filename("registered.nii")
   registered_data = registered.get_fdata()

Alternatively, if you want to co-register many images to the same target, you can do:

.. code-block:: python

   # setup up registration
   from intensity_normalization.util.coregister import Registrator
   transformation = "Affine"
   interpolator = "bSpline"
   initial_rigid = True

   registrator = Registrator(
       target,
       type_of_transform=transformation,
       interpolator=interpolator,
       initial_rigid=initial_rigid
   )

   registered = registrator(image)
   registered.to_filename("registered.nii")
   registered_data = registered.get_fdata()

   # or if you have many images
   images = [nib.load(path_to_image) for path_to_image in image_paths]
   registered_images = registrator.register_images(images)

Saving fit information for sample-based methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fitting and using the resultant fit for new images is supported in the Python API. For example, you can run:

.. code-block:: python

   # load images
   import nibabel as nib
   image_paths = ["path/to/image1.nii", "path/to/image2.nii", ...]
   images = [nib.load(image_path).get_fdata() for image_path in image_paths]

   # normalize the images and save the standard histogram
   from intensity_normalization.normalize.nyul import NyulNormalize
   nyul_normalizer = NyulNormalize()
   nyul_normalizer.fit(images)
   normalized = [nyul_normalizer(image) for image in images]
   nyul_normalizer.save_standard_histogram("standard_histogram.npy")

   # load new images (of the same modality) and normalize those
   new_image_paths = ["path/to/another/image1.nii", "path/to/another/image2.nii", ...]
   new_images = [nib.load(image_path).get_fdata() for image_path in new_image_paths]
   normalized = [nyul_normalizer(image) for image in images]

   # load the standard histogram
   new_nyul_normalizer = NyulNormalize()
   new_nyul_normalizer.load_standard_histogram("standard_histogram.npy")
   normalized = [new_nyul_normalizer(image) for image in images]

For LSQ:

.. code-block:: python

   from intensity_normalization.normalize.lsq import LSQNormalize
   lsq_normalizer = LSQNormalize()
   lsq_normalizer.fit(images)
   normalized = [lsq_normalizer(image) for image in images]
   lsq_normalizer.save_standard_tissue_means("tissue_means.npy")

   # reload the tissue means and use
   lsq_normalizer = LSQNormalize()
   lsq_normalizer.load_standard_tissue_means("tissue_means.npy")
   normalized = [lsq_normalizer(image) for image in images]

RAVEL is only meant to work on a particular batch, so you need to refit it if you add new data to your batch or want to
use it to normalize new data.

Similar options are added to the CLI. For ``nyul-normalize`` the relevant new options are ``--save-standard-histogram``
and ``--load-standard-histogram``. For LSQ, ``--save-standard-tissue-means`` and ``--load-standard-tissue-means``.

.. [*] If you're on a Mac, ``brew install cmake`` and then ``pip install antspyx`` in the environment you want to
       run ``intensity-normalization`` from or install ``intensity-normalization`` with
       ``pip install "intensity-normalization[ants]"``
