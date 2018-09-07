intensity-normalization
=======================

[![Build Status](https://travis-ci.org/jcreinhold/intensity-normalization.svg?branch=master)](https://travis-ci.org/jcreinhold/intensity-normalization)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/intensity-normalization/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/intensity-normalization?branch=master)
[![Documentation Status](https://readthedocs.org/projects/intensity-normalization/badge/?version=latest)](http://intensity-normalization.readthedocs.io/en/latest/?badge=latest)
[![Docker Automated Build](https://img.shields.io/docker/build/jcreinhold/intensity-normalization.svg)](https://hub.docker.com/r/jcreinhold/intensity-normalization/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

This package contains various routines to normalize the intensity of various contrasts of magnetic resonance (MR) brain images; specifically, 
T1-weighted (T1-w), T2-weighted (T2-w), and FLuid-Attenuated Inversion Recovery (FLAIR). Intensity normalization is an important pre-processing step
in many image processing applications regarding MR images since MR images do not have a consistent intensity scale. That is,
even when a single scanner is run with the same parameters and pulse sequence, the intensity scale of the output images may not 
be comparable. We implement various individual image-based and sample-based (i.e., a set of images of the same contrast taken from the same scanner) 
intensity normalization routines to help alleviate this issue.

We implement the following normalization methods:

- Z-score normalization
- Fuzzy C-means (FCM)-segmentation-based white matter (WM) mean normalization
- Gaussian Mixture Model (GMM)-based WM mean normalization
- Kernel Density Estimate WM Peak normalization
- Piecewise Linear Histogram Matching [1,2]
- WhiteStripe [3]
- RAVEL [4]

** Note that while this release was carefully inspected, there may be bugs. Please submit an issue if you encounter a problem. **

This package was developed by [Jacob Reinhold](https:jcreinhold.github.io) and the other students and researchers of the 
[Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

[Link to main Gitlab Repository](https://gitlab.com/jcreinhold/intensity-normalization)

Requirements
------------

- antspy
- matplotlib
- numpy
- nibabel
- scikit-fuzzy
- scikit-learn
- scipy
- statsmodels

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

where `t1/` is a directory full of N T1-w images and `masks/` is a directory full of N corresponding brain masks,
`test_ravel` is the output directory for the normalized images, and `-v` controls the verbosity of the output. 

The command line interface is standard across all normalization routines (i.e., you should be able to 
run all normalization routines with the same call as in the above example), however each has unique options.
Call any executable script with the `-h` flag to see more detailed instructions about the proper call.

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v --with-coverage --cover-tests --cover-package=intensity_normalization tests

If you are using docker, then the equivalent command will be (depending on how the image was built):

    docker run jcreinhold/intensity-normalization /bin/bash -c "pip install nose && nosetests -v tests/"

Singularity
-----------

You can build a singularity image from the docker image hosted on dockerhub via the following command:

    singularity pull --name intensity_normalization.simg docker://jcreinhold/intensity-normalization
    
References
----------

[1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081, 1999.

[2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold, D. L. Collins, and T. Arbel, “Evaluating intensity
    normalization on MRIs of human brain with multiple sclerosis,” Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.
    
[3] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee, F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
    D. S. Reich, and C. M. Crainiceanu, “Statistical normalization techniques for magnetic resonance imaging,” NeuroImage Clin., vol. 6, pp. 9–19, 2014.
    
[4] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu, and R. T. Shinohara, “Removing inter-subject technical variability
    in magnetic resonance imaging studies,” NeuroImage, vol. 132, pp. 198–212, 2016.
