#!/usr/bin/env bash
#
# use the following command to run this script: . ./create_env.sh
#
# use `--antspy` flag to install with antspy
#
# Created on: Apr 27, 2018
# Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

ANTSPY=false

if [[ "$1" == "--antspy" ]]; then
  ANTSPY=true
fi

if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
    :
else
    echo "Operating system must be either linux or OS X"
    return 1
fi

command -v conda >/dev/null 2>&1 || { echo >&2 "I require anaconda but it's not installed. Aborting."; return 1; }

# first make sure conda is up-to-date
conda update -n base conda --yes

packages=(
    coverage
    libiconv
    matplotlib=3.2.2
    mock
    nose
    numpy=1.18.5
    pandas=1.0.5
    pillow=7.2.0
    retrying
    scikit-image=0.16.2
    scikit-learn=0.23.1
    scipy=1.5.0
    sphinx
)

conda_forge_packages=(
    chart-studio
    nibabel=3.1.1
    plotly
    sphinx-argparse
    statsmodels=0.11.1
    webcolors
    scikit-fuzzy=0.4.2
)

conda create --override-channels -c defaults -n intensity_normalization python=3.8 "${packages[@]}" -y
source activate intensity_normalization
conda install -c conda-forge "${conda_forge_packages[@]}" -y

if $ANTSPY; then
    pip install antspyx=0.2.4
    python setup.py install --antspy --preprocess
else
    python setup.py install --preprocess
fi

echo "intensity_normalization conda env script finished"
