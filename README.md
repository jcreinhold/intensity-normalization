intensity-normalization
=====

<!--[![Build Status](https://travis-ci.org/jcreinhold/psola.svg?branch=master)](https://travis-ci.org/jcreinhold/psola)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/psola/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/psola?branch=master)
[![Code Climate](https://codeclimate.com/github/jcreinhold/psola/badges/gpa.svg)](https://codeclimate.com/github/jcreinhold/psola)
[![Issue Count](https://codeclimate.com/github/jcreinhold/psola/badges/issue_count.svg)](https://codeclimate.com/github/jcreinhold/psola)-->

This package contains code to normalize the intensity of various contrasts of MR neuro images

Requirements
------------
- Numpy
- Matplotlib
- Scipy
- pydicom
- nibabel
- dipy
- nipype>=0.12.1
- statsmodels
- scikit-learn
- scikit-fuzzy

Basic Usage
-----------

Install from the source directory

    python setup.py install
    
or (if you actively want to make changes to the package)

    python setup.py develop

and use the command line script to interface with the package

    normalize_intensity ...
    
Project Structure
-----------------
```
intensity_normalization
│
└───intensity_normalization (source code)
│   │   errors.py (project specific exceptions)
│   │   
│   └───exec (holds executables)
│   │   │   normalize_intensity (master script to run any particular normalization routine)
│   │   
│   └───normalize (modules for doing the actual intensity normalization)
│   │   │   fcm (use fuzzy c-means method for WM normalization)
│   │   │   gmm (use gaussian mixture model method for WM normalization)
│   │   │   kde (use kernel density estimate method for WM normalization)
│   │
│   └───utilities (functions not explicitly part of any particular intensity normalization routine)
│
└───tests (unit tests)
│   
└───docs (documentation)
```
