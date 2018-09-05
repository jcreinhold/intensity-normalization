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
    matplotlib==2.2.2
    nose
    numpy==1.15.1
    pandas==0.23.4
    pillow==5.2.0
    scikit-learn==0.19.2
    scikit-image==0.14.0
    scipy==1.1.0
    sphinx
    vtk==8.1.1
)

conda_forge_packages=(
    sphinx-argparse
    itk==4.13.1
    libiconv
    nibabel==2.3.0
    plotly==3.1.1
    statsmodels==0.9.0
    webcolors==1.8.1
)

conda create --channel conda-forge --name intensity_normalization ${packages[@]} ${conda_forge_packages[@]} --yes || return
source activate intensity_normalization || return
pip install -U scikit-fuzzy==0.3.1 || return

# install ANTsPy
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl
else
    # the wheel for OS X appears to be broken, need to build from source
    git clone https://github.com/ANTsX/ANTsPy.git
    cd ANTsPy
    git checkout v0.1.5
    python setup.py develop
    cd ..
fi

# now finally install the intensity-normalization package
python setup.py install || return

echo "intensity_normalization conda env script finished (verify yourself if everything installed correctly)"
