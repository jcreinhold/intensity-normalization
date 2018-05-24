#!/bin/bash
# use the following command to run this script: . ./create_intensity_normalization_env.sh
# since we download a git directory and build it from source
# make sure you are in a directory where you are ok with downloading a folder 
# that you *cannot* remove if you want the packages to work (on a Mac)
# Created on: Apr 27, 2018
# Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
    :
else
    echo "Operating system must be either linux or OS X"
    return 1
fi

command -v conda >/dev/null 2>&1 || { echo >&2 "I require anaconda but it's not installed.  Aborting."; return 1; }
command -v flirt >/dev/null 2>&1 || { echo >&2 "I require FSL but it's not installed.  Aborting."; return 1; }
command -v cmake >/dev/null 2>&1 || { echo >&2 "I require cmake but it's not installed.  Aborting."; return 1; }

# first make sure conda is up-to-date
conda update -n base conda --yes

packages=(
    numpy 
    matplotlib 
    scipy 
    seaborn 
    scikit-learn 
    nose 
    mock 
    sphinx
    coverage
    vtk
)

conda_forge_packages=(
    nibabel 
    rpy2 
    r-essentials
    r-mgcv 
    r-mnormt 
    r-nlme 
    r-psych
    r-git2r
    webcolors
    plotly
    libiconv
    itk
    sphinx-argparse
)

conda create --name intensity_normalization ${packages[@]} --yes
source activate intensity_normalization 
pip install -U scikit-fuzzy
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    pip install antspy
else 
    git clone https://github.com/ANTsX/ANTsPy.git
    cd ANTsPy
    python setup.py develop
    cd ..
fi
conda install -c conda-forge ${conda_forge_packages[@]} --yes
conda env export > intensity_normalization_env.yml

# handle installing appropriate r packages (assumes FSL is installed)
# add default server to Rprofile so that install package scripts run w/o error
chooseserver='options(repos=structure(c(CRAN="http://cran.us.r-project.org")))'
if [ -f "~/.Rprofile" ]; then
    if grep -q $chooseserver "~/.Rprofile"; then
        :
    else
        echo $chooseserver >> ~/.Rprofile
    fi
else
    echo $chooseserver > ~/.Rprofile
fi

R -e 'packages = installed.packages(); packages = packages[, "Package"]; if (!"devtools" %in% packages) { install.packages("devtools") };'
R -e 'source("https://neuroconductor.org/neurocLite.R"); neuroc_install(c("ITKR", "ANTsR"));'
R -e 'source("https://neuroconductor.org/neurocLite.R"); neuro_install(c("fslr", "neurobase"));'
R -e 'packages = installed.packages(); packages = packages[, "Package"]; if (!"limma" %in% packages) { source("https://bioconductor.org/biocLite.R"); biocLite("limma") };'
R -e 'source("https://neuroconductor.org/neurocLite.R"); neuro_install(c("WhiteStripe", "RAVEL", "robex"))'

echo "intensity_normalization conda env script finished (verify yourself if everything installed correctly)"
