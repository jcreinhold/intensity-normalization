intensity-normalization
=====

This package contains code to normalize the intensity of various contrasts of MR neuro images

Requirements
------------

- antspy
- matplotlib
- numpy
- nibabel
- scikit-fuzzy
- scikit-learn

We have provided a script `create_env.sh` to create a conda environment with the necessary packages 
(run like: `. ./create_env.sh`, this package will be installed in the created environment)

To use ROBEX, R and the rpy2 python package need to be installed along with robex in R (this is not handled in 
the environment script).

Basic Usage
-----------

Install from the source directory

    python setup.py install
    
or (if you actively want to make changes to the package)

    python setup.py develop

and use the several provided command line scripts to interface with the package,
e.g., 

    ravel-normalize -i t1/ -m masks/ -o test_ravel -v

the command line interface is standard across all normalization routines (i.e., you should be able to 
run all normalization routines with the same call as in the above example), however each has unique options.
Call any executable script with the `-h` flag to see more detailed instructions about the proper call.

Project Structure
-----------------
```
intensity_normalization
│
└───intensity_normalization (source code)
│   │   errors.py (project specific exceptions)
│   │   
│   └───exec (holds executables)
│   │   │   zscore-normalize (call Z-score normalization on an directory)
│   │   │   fcm-normalize (call fuzzy c-means WM normalization on an directory)
│   │   │   gmm-normalize (call GMM WM normalization on an directory)
│   │   │   kde-normalize (call KDE WM normalization on an directory)
│   │   │   hm-normalize (call N&U HM intensity normalization on an directory)
│   │   │   ws-normalize (call WhiteStripe WM normalization on an directory)
│   │   │   ravel-normalize (call RAVEL intensity normalization on an directory)
│   │   │   robex (create brain masks for directory of images w/ ROBEX)
│   │   │   preprocess (resample, reorient, and bias field correct dir of imgs)
│   │   │   plot-hists (CLI to plot histograms for a directory)
│   │   │   tissue-mask (CLI to create tissue masks (i.e., CSF/GM/WM for a directory)
│   │   
│   └───normalize (modules for doing the actual intensity normalization)
│   │   │   zscore (use Z-score method for intensity normalization)
│   │   │   fcm (use fuzzy c-means method for WM normalization)
│   │   │   gmm (use gaussian mixture model method for WM normalization)
│   │   │   kde (use kernel density estimate method for WM normalization)
│   │   │   hm (use Nyul & Udapa method for intensity normalization)
│   │   │   ws (use WhiteStripe method for WM normalization)
│   │   │   ravel (use RAVEL method for intensity normalization)
│   │
│   └───plot (modules for visualizing results of normalization)
│   │   |   hists (functions to plot histograms for all nifti images in a directory)
│   │   
│   └───utilities (functions not explicitly part of any particular intensity normalization routine)
│
└───tests (unit tests)
│   
└───docs (documentation)
```

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v --with-coverage --cover-tests --cover-package=intensity_normalization tests
