#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
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

extras_requirements = {
    "ants": ["antspyx"],
}

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
    entry_points={
        "console_scripts": [
            "fcm-normalize=intensity_normalization.cli:fcm_main",
            "kde-normalize=intensity_normalization.cli:kde_main",
            "lsq-normalize=intensity_normalization.cli:lsq_main",
            "nyul-normalize=intensity_normalization.cli:nyul_main",
            "ravel-normalize=intensity_normalization.cli:ravel_main",
            "ws-normalize=intensity_normalization.cli:ws_main",
            "zscore-normalize=intensity_normalization.cli:zscore_main",
            "plot-histograms=intensity_normalization.cli:histogram_main",
            "coregister=intensity_normalization.cli:register_main",
        ],
    },
    install_requires=requirements,
    extras_require=extras_requirements,
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
