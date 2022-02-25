#!/usr/bin/env bash
# Run all intensity-normalization CLIs on a directory of images
# Author: Jacob Reinhold <jcreinhold@gmail.com>

dry_run=false
ext="nii.gz"
modality="t1"
verbosity=1
all_tissue_memberships_dir=""
tissue_memberships_dir=""
csf_masks_dir=""

usage="$(basename "$0") [-h] [-r] [-d str] [-e str] [-m str] [-a str] [-t str] [-c str] [-v int] -- Run all CLIs on a directory of images
where:
    -h  show this help text
    -d  path to directory of skull-stripped MR images (required)
    -e  extension of the data (default: ${ext})
    -m  modality of the images in the specified directory (default: ${modality})
    -a  path to directory containing all tissue memberships (required when modality != 't1')
    -t  path to directory containing one tissue membership (e.g., WM) (required when modality != 't1')
    -c  path to directory containing CSF masks (required when modality != 't1')
    -r  do a dry run, i.e., run this script but skip running all CLIs (for debugging this script)
    -v  verbosity level of CLIs (default: ${verbosity})"

while getopts ':hd:e:m:a:t:c:v::r' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    a) all_tissue_memberships_dir=$OPTARG
       ;;
    c) csf_masks_dir=$OPTARG
       ;;
    d) directory=$OPTARG
       ;;
    e) ext=$OPTARG
       ;;
    m) modality=$OPTARG
       ;;
    t) tissue_memberships_dir=$OPTARG
       ;;
    r) dry_run=true
       ;;
    v) verbosity=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

CYAN='\033[0;36m'
GREEN='\033[0;32m'
LIGHT_CYAN='\033[1;36m'
LIGHT_PURPLE='\033[0;35m'
NC='\033[0m'  # No Color
PURPLE='\033[1;35m'
RED='\033[0;31m'
YELLOW='\033[1;33m'

if [ -z "${directory}" ]
then
    echo >&2 -e "${RED}'directory' is a required argument.${NC}"
    exit 1
fi

if [ "${modality}" != "t1" ] && [ -z "${all_tissue_memberships_dir}" ]
then
    echo >&2 -e "${RED}If '-m' != 't1', then '-a', '-t', '-c' all required.${NC}"
    exit 1
fi

if [ "${modality}" != "t1" ] && [ -z "${tissue_memberships_dir}" ]
then
    echo >&2 -e "${RED}If '-m' != 't1', then '-a', '-t', '-c' all required.${NC}"
    exit 1
elif [ -n "${tissue_memberships_dir}" ]
then
    tm_fns=("${tissue_memberships_dir}"/*."${ext}")
fi

if [ "${modality}" != "t1" ] && [ -z "${csf_masks_dir}" ]
then
    echo >&2 -e "${RED}If '-m' != 't1', then '-a', '-t', '-c' all required.${NC}"
    exit 1
fi

verbosity_flag=$(python -c "print('-' + ('v' * ${verbosity}))")

# choose a CLI that's always installed to test if intensity-normalization installed
fcm-normalize --version \
|| { echo >&2 -e "${RED}intensity-normalization is not installed.${NC}"; exit 1; }

# verify that antspyx is installed
ravel-normalize --version >/dev/null 2>&1 \
|| { echo >&2 -e "${RED}intensity-normalization needs to be installed with 'ants' extras.${NC}"; exit 1; }

# verify that matplotlib is installed
plot-histograms --version >/dev/null 2>&1 \
|| { echo >&2 -e "${RED}intensity-normalization needs to be installed with 'plot' extras.${NC}"; exit 1; }

echo ""

single_image_clis=(
    "fcm-normalize fcm"
    "kde-normalize kde"
    "ws-normalize whitestripe"
    "zscore-normalize zscore"
    "tissue-membership tissue"
    "preprocess preprocessed"
    "coregister coregistered"
)

sample_based_clis=(
    "lsq-normalize lsq"
    "nyul-normalize nyul"
    "ravel-normalize ravel"
)

echo "Plotting histograms for input directory"
if ! "${dry_run}"
then
    plot-histograms "${directory}" "${verbosity_flag}" \
    || { echo >&2 -e "${RED}Histogram plotter failed. Exiting.${NC}"; exit 1; }
fi
echo ""

for i in "${single_image_clis[@]}"
do
    # shellcheck disable=SC2086  # word splitting is intentional
    set -- $i
    cli="${1}"
    output_directory="${2}"
    mkdir -p "${output_directory}"
    echo -e "Running ${CYAN}${cli}${NC} CLI"
    count=0
    for image_fn in "${directory}"/*."${ext}"
    do
        echo -e "Processing image: ${PURPLE}${image_fn}${NC}"
        fn=$(basename "${image_fn}")
        if [ "${modality}" == 't1' ] || [ "${output_directory}" != "fcm" ]
        then
            if ! "${dry_run}"
            then
                "${cli}" \
                    "${image_fn}" \
                    -o "${output_directory}/${fn}" \
                    "${verbosity_flag}" \
                || { echo >&2 -e "${RED}${cli} failed. Exiting.${NC}"; exit 1; }
            fi
        elif [ "${output_directory}" == "fcm" ]
        then
            tm_fn="${tm_fns[count]}"
            echo -e "Tissue membership: ${LIGHT_PURPLE}${tm_fn}${NC}"
            if ! "${dry_run}"
            then
                "${cli}" \
                    "${image_fn}" \
                    -o "${output_directory}/${fn}" \
                    -mo "${modality}" \
                    -tm "${tm_fn}" \
                    "${verbosity_flag}" \
                || { echo >&2 -e "${RED}${cli} failed. Exiting.${NC}"; exit 1; }
            fi
        else
            echo >&2 -e "${RED}Configuration for ${cli} not implemented. Exiting.${NC}"
            exit 1
        fi
        (( ++count ))
    done
    echo -e "Plotting histograms of results for ${CYAN}${cli}${NC}"
    if ! "${dry_run}"
    then
        plot-histograms "${output_directory}" "${verbosity_flag}" \
        || { echo >&2 -e "${RED}Histogram plotter failed. Exiting.${NC}"; exit 1; }
    fi
    echo -e "${CYAN}${cli}${NC} results saved to ${LIGHT_CYAN}${output_directory}${NC}"
    echo ""
done

for i in "${sample_based_clis[@]}"
do
    # shellcheck disable=SC2086
    set -- $i
    cli="${1}"
    output_directory="${2}"
    mkdir -p "${output_directory}"
    echo -e "Running ${CYAN}${cli}${NC} CLI"
    if [ "${modality}" == 't1' ] || [ "${output_directory}" == "nyul" ]
    then
        if ! "${dry_run}"
        then
            "${cli}" \
                "${directory}" \
                -o "${output_directory}" \
                -p \
                -e "${ext}" \
                "${verbosity_flag}" \
            || { echo >&2 -e "${RED}${cli} failed. Exiting.${NC}"; exit 1; }
        fi
    elif [ "${output_directory}" == "lsq" ]
    then
        if ! "${dry_run}"
        then
            "${cli}" \
                "${directory}" \
                -o "${output_directory}" \
                -mo "${modality}" \
                -tm "${all_tissue_memberships_dir}" \
                -p \
                -e "${ext}" \
                "${verbosity_flag}" \
            || { echo >&2 -e "${RED}${cli} failed. Exiting.${NC}"; exit 1; }
        fi
    elif [ "${output_directory}" == "ravel" ]
    then
        if ! "${dry_run}"
        then
            "${cli}" \
                "${directory}" \
                -o "${output_directory}" \
                -mo "${modality}" \
                -m "${csf_masks_dir}" \
                -p \
                -e "${ext}" \
                --masks-are-csf \
                --no-registration \
                "${verbosity_flag}" \
            || { echo >&2 -e "${RED}${cli} failed. Exiting.${NC}"; exit 1; }
        fi
    else
        echo >&2 -e "${RED}Configuration for ${cli} not implemented. Exiting.${NC}"
        exit 1
    fi
    echo -e "${CYAN}${cli}${NC} results saved to ${LIGHT_CYAN}${output_directory}${NC}"
    echo ""
done

echo ""
echo -e "${GREEN}All CLIs ran successfully.${NC}"
echo -e "${YELLOW}Verify the histograms and images to ensure correct functionality.${NC}"
