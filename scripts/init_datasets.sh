# |-------------------------------------|
# |       [CM3070] FINAL PROJECT        |
# |    CLASSIFYING SATELLITE IMAGERY    |
# |    BSc CS - UNIVERSITY OF LONDON    |
# |             ARJUN BAJAJ             |
# |-------------------------------------|


# SETUP SCRIPT
# --------------------
# Note: run this script in a conda environment with all the required packages installed.
# See project readme for setup instructions!


# Path for the current directory of the script
CURRENT_DIR=$(dirname $(realpath -s "$0"))

# Path for the `datasets` directory in the repo
DATASETS="$CURRENT_DIR/../datasets/"

# Delete the `datasets` directory and the temporary directories
rm -irf $DATASETS
rm -irf /tmp/eurosat
rm -irf /tmp/uc_merced

# Ensure directory to cache original dataset downloads exists
mkdir $CURRENT_DIR/../.cache

# Create the `datasets` directory in the repo
mkdir $DATASETS

# EuroSAT Dataset
# -----------------------
# The EuroSAT dataset is my primary dataset for the Final Project.

# Download the EuroSAT dataset file
wget -nc https://madm.dfki.de/files/sentinel/EuroSAT.zip -O $CURRENT_DIR/../.cache/eurosat.zip

# Unzip the dataset
unzip -q -n -x -d /tmp/eurosat_tmp $CURRENT_DIR/../.cache/eurosat.zip

# Move the inner directory to a tmp location
mv /tmp/eurosat_tmp/2750/ /tmp/eurosat

# Create the dataset directory inside the repo
mkdir "$DATASETS/eurosat"

# Split the dataset into 50% training, 25% validation, and 25% testing images
mkdir $DATASETS/eurosat-full
cp -R /tmp/eurosat $DATASETS/eurosat-full
python3 $CURRENT_DIR/split_dataset.py /tmp/eurosat $DATASETS/eurosat --train 0.8 --val 0.1

# This script prepares test images to perform inference on the Raspberry Pi Pico.
python3 $CURRENT_DIR/prep_images_for_pico.py


# UC Merced Dataset
# -----------------------
# The UC Merced Dataset is used to perform transfer learning from the EuroSAT Dataset.

# Download the UC Merced dataset file
wget -nc http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip -O $CURRENT_DIR/../.cache/uc_merced.zip

# Unzip the dataset
unzip -q -n -x -d /tmp/uc_merced_tmp $CURRENT_DIR/../.cache/uc_merced.zip

# Create a tmp directory
mkdir /tmp/uc_merced

# Change permissions on the entire directory recursively,
# otherwise it fails to move the inner directories
chmod -R 0777 /tmp/uc_merced_tmp

# Move the inner directory to a tmp location
mv /tmp/uc_merced_tmp/UCMerced_LandUse/Images/* /tmp/uc_merced/

# Convert all TIF files to JPG,
# as Tensorflow does not yet support reading TIF files directly.
cd /tmp/uc_merced
for i in *; do mogrify -format jpg "$i/*.tif"; done

# Delete all TIF files
rm -rf **/*.tif

# Go back to the current script directory
cd $CURRENT_DIR

# Create the dataset directory inside the repo
mkdir "$DATASETS/uc_merced"

# Split the dataset into 50% training, 25% validation, and 25% testing images
python3 $CURRENT_DIR/split_dataset.py /tmp/uc_merced $DATASETS/uc_merced --train 0.8 --val 0.1
