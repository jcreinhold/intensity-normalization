#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "matplotlib",
    "nibabel",
    "numpy",
    "scikit-image",
    "scikit-learn",
    "scikit-fuzzy",
    "scipy",
    "statsmodels",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Jacob Reinhold",
    author_email="jcreinhold@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="normalize the intensities of various MR image modalities",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="intensity normalization",
    name="intensity-normalization",
    packages=find_packages(
        include=["intensity_normalization", "intensity_normalization.*"]
    ),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jcreinhold/intensity-normalization",
    version="2.0.0",
    zip_safe=False,
)
