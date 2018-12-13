#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs intensity-normalization package
Can be run via command: python setup.py install (or develop)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 24, 2018
"""

from setuptools import setup, find_packages
from sys import platform
import sys
import warnings


install_antspy = '--antspy' in sys.argv[1:]

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

if install_antspy:
    warnings.warn('Will try to install antspy. There is a reasonable chance that this will not work. Build antspy from source if that is the case.')
    if platform == "linux" or platform == "linux32":
        if '--1.4' in sys.argv[1:]:
            antspy = "https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl"
        else:
            antspy = "https://github.com/ANTsX/ANTsPy/releases/download/v0.1.6/antspy-0.1.6-cp36-cp36m-linux_x86_64.whl"
    elif platform == "darwin":
        antspy = "https://github.com/ANTsX/ANTsPy/releases/download/v0.1.6/antspy-0.1.6-cp36-cp36m-macosx_10_13_x86_64.whl"
    else:
        raise Exception('antspy package only supports linux and OS X, must install without antspy option')
else:
    antspy = ""

args = dict(
    name='intensity-normalization',
    version='1.2.1',
    description="Normalize the intensity values of MR images",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/intensity-normalization',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'tutorials')),
    entry_points = {
        'console_scripts': ['fcm-normalize=intensity_normalization.exec.fcm_normalize:main',
                            'gmm-normalize=intensity_normalization.exec.gmm_normalize:main',
                            'hm-normalize=intensity_normalization.exec.hm_normalize:main',
                            'kde-normalize=intensity_normalization.exec.kde_normalize:main',
                            'ravel-normalize=intensity_normalization.exec.ravel_normalize:main',
                            'ws-normalize=intensity_normalization.exec.ws_normalize:main',
                            'zscore-normalize=intensity_normalization.exec.zscore_normalize:main',
                            'robex=intensity_normalization.exec.robex:main',
                            'preprocess=intensity_normalization.exec.preprocess:main',
                            'plot-hists=intensity_normalization.exec.plot_hists:main',
                            'norm-quality=intensity_normalization.exec.norm_quality:main',
                            'coregister=intensity_normalization.exec.coregister:main',
                            'tissue-mask=intensity_normalization.exec.tissue_mask:main']
    },
    keywords="mr intensity normalization",
    dependency_links=[antspy]
)

setup(install_requires=['matplotlib',
                        'nibabel',
                        'numpy',
                        'scikit-image',
                        'scikit-learn',
                        'scikit-fuzzy',
                        'scipy',
                        'statsmodels'] + ['antspy'] if install_antspy else [], **args)
