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

command -v conda >/dev/null 2>&1 || { echo >&2 "I require anaconda but it's not installed.  Aborting."; return 1; }

# first make sure conda is up-to-date
conda update -n base conda --yes

packages=(
    coverage
    libiconv
    matplotlib=3.1.1
    nose
    numpy=1.16.5
    pandas=0.25.1
    pillow=6.1.0
    scikit-learn=0.21.2
    scikit-image=0.15.0
    scipy=1.3.1
    sphinx
)

conda_forge_packages=(
    nibabel=2.5.1
    sphinx-argparse
    statsmodels=0.10.1
    webcolors=1.9.1
    scikit-fuzzy==0.4.2
)

conda create --override-channels -c defaults -n intensity_normalization python=3.7 ${packages[@]} -y
source activate intensity_normalization
conda install -c conda-forge ${conda_forge_packages[@]} -y

if $ANTSPY; then
    pip install antspyx
    python setup.py install --antspy --preprocess
else
    python setup.py install --preprocess
fi

# now finally install the intensity-normalization package
echo "intensity_normalization conda env script finished (verify yourself if everything installed correctly)"
