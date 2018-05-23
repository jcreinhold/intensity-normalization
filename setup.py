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


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='intensity-normalization',
    version='0.0.0',
    description="Normalize the intensity values of MR images",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/intensity-normalization',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    scripts=['intensity_normalization/exec/fcm-normalize',
             'intensity_normalization/exec/gmm-normalize',
             'intensity_normalization/exec/hm-normalize',
             'intensity_normalization/exec/kde-normalize',
             'intensity_normalization/exec/ravel-normalize',
             'intensity_normalization/exec/ws-normalize',
             'intensity_normalization/exec/robex',
             'intensity_normalization/exec/preprocess',
             'intensity_normalization/exec/plot-hists',
             'intensity_normalization/exec/tissue-mask'],
    keywords="mr intensity normalization"
)

setup(install_requires=['numpy',
                        'scipy',
                        'pydicom',
                        'nibabel',
                        'dipy',
                        'nipype>=0.12.1',
                        'matplotlib',
                        'statsmodels',
                        'scikit-learn',
                        'scikit-fuzzy',
                        'rpy2'], **args)
