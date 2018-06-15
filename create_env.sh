#!/usr/bin/env bash
#
# use the following command to run this script: . ./create_env.sh
#
# Created on: Apr 27, 2018
# Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
    :
else
    echo "Operating system must be either linux or OS X"
    return 1
fi

command -v conda >/dev/null 2>&1 || { echo >&2 "I require anaconda but it's not installed.  Aborting."; return 1; }

# first make sure conda is up-to-date
conda update -n base conda --yes || return

packages=(
    coverage
    matplotlib
    nose
    numpy
    pillow
    scikit-learn
    scikit-image
    scipy
    seaborn
    sphinx
    vtk
)

conda_forge_packages=(
    itk
    libiconv
    nibabel
    plotly
    sphinx-argparse
    statsmodels
    webcolors
)

conda create --channel conda-forge --name intensity_normalization ${packages[@]} ${conda_forge_packages[@]} --yes || return
source activate intensity_normalization || return
pip install -U scikit-fuzzy || return
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl
else
    pip install https://github.com/ANTsX/ANTsPy/releases/download/Weekly/antspy-0.1.4-cp36-cp36m-macosx_10_7_x86_64.whl
fi

# now finally install the intensity-normalization package
python setup.py install || return

echo "intensity_normalization conda env script finished (verify yourself if everything installed correctly)"
