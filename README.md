intensity-normalization
=======================

[![Build Status](https://travis-ci.org/jcreinhold/intensity-normalization.svg?branch=master)](https://travis-ci.org/jcreinhold/intensity-normalization)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/intensity-normalization/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/intensity-normalization?branch=master)
[![Documentation Status](https://readthedocs.org/projects/intensity-normalization/badge/?version=latest)](http://intensity-normalization.readthedocs.io/en/latest/)
[![Docker Automated Build](https://img.shields.io/docker/build/jcreinhold/intensity-normalization.svg)](https://hub.docker.com/r/jcreinhold/intensity-normalization/)
[![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

This package contains various routines to normalize the intensity of various contrasts of magnetic resonance (MR) brain images; specifically, 
T1-weighted (T1-w), T2-weighted (T2-w), and FLuid-Attenuated Inversion Recovery (FLAIR). Intensity normalization is an important pre-processing step
in many image processing applications regarding MR images since MR images do not have a consistent intensity scale. We implement various individual 
image-based and sample-based (i.e., a set of images of the same contrast taken from the same scanner) 
intensity normalization routines to help alleviate this issue.

Note that this package was developed to process *adult human* MR images; neonatal, pediatric, and animal MR images should 
also work (at least with some of the normalization schemes). But if there is a substantial difference in the contrast 
you are processing compared with the same contrast in an adult human, the normalization scheme may not work as expected.
Please open an issue if you encounter such a problem.

Also note that when the package refers to any specific contrast image, it is referring to a *non-contrast* version unless 
otherwise stated. Using a contrast image as input to a method that assumes non-contrast will produce suboptimal results.

We implement the following normalization methods:

- Z-score normalization
- Fuzzy C-means (FCM)-segmentation-based white matter (WM) mean normalization
- Least squares (LSQ) tissue mean normalization
- Gaussian Mixture Model (GMM)-based WM mean normalization
- Kernel Density Estimate WM Peak normalization
- Piecewise Linear Histogram Matching [1,2]
- WhiteStripe [3]
- RAVEL [4]

We use this package to explore the impact of intensity normalization on a synthesis task (pre-print available [here](https://arxiv.org/abs/1812.04652)).

** Note that while this release was carefully inspected, there may be bugs. Please submit an issue if you encounter a problem. **

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io) and the other students and researchers of the 
[Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

[Link to main Gitlab Repository](https://gitlab.com/jcreinhold/intensity-normalization) (The Github repo is a mirror of this Gitlab repo.)

Requirements
------------

- matplotlib
- numpy
- nibabel
- scikit-fuzzy
- scikit-learn
- scipy
- statsmodels

We have provided a script `create_env.sh` to create a conda environment with the necessary packages 
(run like: `. ./create_env.sh`, this package will be installed in the created environment)

Basic Usage
-----------

The easiest way to install the package is through the following command:

    pip install git+git://github.com/jcreinhold/intensity-normalization.git
    
To install from the source directory, use

    python setup.py install
    
or (if you actively want to make changes to the package)

    python setup.py develop
    
and use the several provided command line scripts to interface with the package,
e.g., 

    fcm-normalize -i t1/ -m masks/ -o test_fcm -v

where `t1/` is a directory full of N T1-w images and `masks/` is a directory full of N corresponding brain masks,
`test_fcm` is the output directory for the normalized images, and `-v` controls the verbosity of the output.

Note the package [antspy](https://github.com/ANTsX/ANTsPy) is required for the RAVEL normalization routine, the 
preprocessing tool as well as the co-registration tool, but all other normalization and processing tools work without it. 
To also install the antspy package either append `--antspy` to your call to `setup.py` or `create_env.sh`. 

To use the convenience pre-processing scripts, install the package with the following command:

    python setup.py install --antspy --preprocess
    
or 

    python setup.py develop --antspy --preprocess
 
The command line interface is standard across all normalization routines (i.e., you should be able to 
run all normalization routines with the same call as in the above example); however, each has unique options
and not all methods support single image processing.

Call any executable script with the `-h` flag to see more detailed instructions about the proper call.

Tutorial
--------

[5 minute Overview](https://github.com/jcreinhold/intensity-normalization/blob/master/tutorials/5min_tutorial.md)

In addition to the above small tutorial, there is consolidated documentation [here](https://intensity-normalization.readthedocs.io/en/latest/).

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

Citation
--------

If you use the `intensity-normalization` package in an academic paper, please cite the corresponding [paper](https://arxiv.org/abs/1812.04652):

    @inproceedings{reinhold2019evaluating,
      title={Evaluating the impact of intensity normalization on {MR} image synthesis},
      author={Reinhold, Jacob C and Dewey, Blake E and Carass, Aaron and Prince, Jerry L},
      booktitle={Medical Imaging 2019: Image Processing},
      volume={10949},
      pages={109493H},
      year={2019},
      organization={International Society for Optics and Photonics}
    }
        
References
----------

[1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081, 1999.

[2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold, D. L. Collins, and T. Arbel, “Evaluating intensity
    normalization on MRIs of human brain with multiple sclerosis,” Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.
    
[3] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee, F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
    D. S. Reich, and C. M. Crainiceanu, “Statistical normalization techniques for magnetic resonance imaging,” NeuroImage Clin., vol. 6, pp. 9–19, 2014.
    
[4] J. P. Fortin, E. M. Sweeney, J. Muschelli, C. M. Crainiceanu, and R. T. Shinohara, “Removing inter-subject technical variability
    in magnetic resonance imaging studies,” NeuroImage, vol. 132, pp. 198–212, 2016.
