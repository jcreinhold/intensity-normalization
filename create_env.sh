#!/usr/bin/env bash
#
# use the following command to run this script: . ./create_env.sh
#
# use `--antspy` flag to install with antspy
# append the `--1.4` flag to install with antspy v0.1.4 instead of v0.1.6 (this may work if v0.1.6 does not)
#   the `--1.4` flag only works on linux.
#
# Created on: Apr 27, 2018
# Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

ANTSPY=false

if [[ "$1" == "--antspy" ]]; then
  ANTSPY=true
fi

V1_4=false

if [[ "$2" == "--1.4" ]]; then
  V1_4=true
fi

if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
    :
else
    echo "Operating system must be either linux or OS X"
    return 1
fi

command -v conda >/dev/null 2>&1 || { echo >&2 "I require anaconda but it's not installed.  Aborting."; return 1; }

# first make sure conda is up-to-date
conda update -n base conda --yes

packages=(
    coverage
    libiconv
    matplotlib=3.0.2
    nose
    numpy=1.15.4
    pandas=0.23.4
    pillow=5.3.0
    scikit-learn=0.20.1
    scikit-image=0.14.1
    scipy=1.1.0
    sphinx
)

conda_forge_packages=(
    nibabel=2.3.0
    sphinx-argparse
    statsmodels=0.9.0
    webcolors=1.8.1
)

conda create --override-channels -c defaults -n intensity_normalization python=3.6.7 ${packages[@]} -y
source activate intensity_normalization
conda install -c conda-forge ${conda_forge_packages[@]} -y
pip install -U scikit-fuzzy==0.4.0

if $ANTSPY; then
    # install ANTsPy
    if [[ "$OSTYPE" == "linux-gnu" ]]; then
        if $V1_4; then
            pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl
        else
            pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.6/antspy-0.1.6-cp36-cp36m-linux_x86_64.whl
        fi
    else
        pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.6/antspy-0.1.6-cp36-cp36m-macosx_10_13_x86_64.whl
    fi
    python setup.py install --antspy --preprocess --quality
else
    python setup.py install --preprocess --quality
fi

# now finally install the intensity-normalization package

echo "intensity_normalization conda env script finished (verify yourself if everything installed correctly)"
