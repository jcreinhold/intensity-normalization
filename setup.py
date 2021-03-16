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

custom_args = sys.argv[2:].copy()
del sys.argv[2:]

install_antspy = '--antspy' in custom_args
link_preprocess = '--preprocess' in custom_args

console_scripts = ['fcm-normalize=intensity_normalization.exec.fcm_normalize:main',
                   'gmm-normalize=intensity_normalization.exec.gmm_normalize:main',
                   'nyul-normalize=intensity_normalization.exec.nyul_normalize:main',
                   'kde-normalize=intensity_normalization.exec.kde_normalize:main',
                   'ws-normalize=intensity_normalization.exec.ws_normalize:main',
                   'zscore-normalize=intensity_normalization.exec.zscore_normalize:main',
                   'ravel-normalize=intensity_normalization.exec.ravel_normalize:main',
                   'lsq-normalize=intensity_normalization.exec.lsq_normalize:main',
                   'plot-hists=intensity_normalization.exec.plot_hists:main']

if link_preprocess: console_scripts.extend(['preprocess=intensity_normalization.exec.preprocess:main',
                                            'coregister=intensity_normalization.exec.coregister:main',
                                            'tissue-mask=intensity_normalization.exec.tissue_mask:main',
                                            'norm-quality=intensity_normalization.exec.norm_quality:main'])

with open('README.md', encoding="utf-8") as f:
    readme = f.read()

with open('LICENSE', encoding="utf-8") as f:
    license = f.read()

if platform not in ["linux", "linux32", "darwin"] and install_antspy:
    raise Exception('antspy package only supports linux and OS X, must install without antspy option')

args = dict(
    name='intensity-normalization',
    version='1.4.4',
    description="Normalize the intensity values of MR images",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/intensity-normalization',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'tutorials')),
    entry_points={
        'console_scripts': console_scripts
    },
    keywords="mr intensity normalization",
)

setup(install_requires=['matplotlib',
                        'nibabel',
                        'numpy',
                        'scikit-image',
                        'scikit-learn',
                        'scikit-fuzzy',
                        'scipy',
                        'statsmodels'] + ['antspyx'] if install_antspy else [], **args)
