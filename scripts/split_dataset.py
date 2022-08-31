# |-------------------------------------|
# |       [CM3070] FINAL PROJECT        |
# |    CLASSIFYING SATELLITE IMAGERY    |
# |    BSc CS - UNIVERSITY OF LONDON    |
# |             ARJUN BAJAJ             |
# |-------------------------------------|

# Split an Image Dataset
# -----------------------
# This script splits an image dataset into
# `train`, `val`, and `test` sets. It takes an input folder,
# an output folder, and the percentages for `train` and `val` splits.
#
# USAGE:
# python3 split_dataset.py ~/DL/eurosat/eurosat-rgb ../datasets/eurosat --train 0.5 --val 0.25


import os, shutil, pathlib, random, sys

# NCC-1701 Starship Enterprise
SEED = 1701


def split_image_dataset(ds_path, split_path, max_files=None, train=0.5, val=0.2):
  """
  Take an image dataset path and split it into train, val, and test sets.

  Parameters
  ----------
  ds_path : str
    Path to the image dataset.
  split_path : str
    Path to the output folder.
  max_files : int, optional
    Maximum number of files to use. All files are considered by default.
  train : float, optional
    Percentage of files to use for training. Defaults to 50%.
  val : float, optional
    Percentage of files to use for validation. Defaults to 20%.
  """

  # Get full paths for the input and output folders
  ds_path = pathlib.Path(ds_path).expanduser()
  split_path = pathlib.Path(split_path).expanduser()

  # Delete the output folder if it exists
  shutil.rmtree(split_path, ignore_errors=True)

  # Make the train, val, and test folders
  os.makedirs(split_path/'train', exist_ok=True)
  os.makedirs(split_path/'val', exist_ok=True)
  os.makedirs(split_path/'test', exist_ok=True)

  # Get the list of folders in the dataset (categories)
  for dir in os.listdir(ds_path):
    if dir != '.DS_Store':
      # Make the category folder in the each of the output subfolders
      os.makedirs(split_path/'train'/dir, exist_ok=True)
      os.makedirs(split_path/'val'/dir, exist_ok=True)
      os.makedirs(split_path/'test'/dir, exist_ok=True)

      # Get the list of images in the category
      files = os.listdir(ds_path/dir)

      # Shuffle the files
      random.seed(SEED)
      random.shuffle(files)

      # If max_files is set, subset the files list
      if max_files != None:
        files = files[0:max_files]

      # Split the files into train, val, and test sets
      train_files = files[:int(len(files) * train)]
      val_files = files[int(len(files) * train):int(len(files) * (train+val))]
      test_files = files[int(len(files) * (train+val)):]

      # Copy the images for each set to the appropriate folder

      for file in train_files:
        shutil.copy(ds_path/dir/file, split_path/'train'/dir)

      for file in val_files:
        shutil.copy(ds_path/dir/file, split_path/'val'/dir)

      for file in test_files:
        shutil.copy(ds_path/dir/file, split_path/'test'/dir)

      print('\t', dir, len(train_files), len(val_files), len(test_files))


# This script is run from the command line
if __name__ == '__main__':

  # If the arguments list is less than three items, then print help message
  if len(sys.argv) < 3:
    print('Usage: python3 init_datasets.py <dataset_path> <split_path> --train <train_percent> --val <val_percent> --max-files <max_files>')
    sys.exit(1)

  dataset_path = sys.argv[1]
  split_path = sys.argv[2]

  train = 0.5
  val = 0.2
  max_files = None

  # Parse the command line arguments
  for i in range(3, len(sys.argv)):
    if sys.argv[i] == '--train':
      train = float(sys.argv[i+1])
    elif sys.argv[i] == '--val':
      val = float(sys.argv[i+1])
    elif sys.argv[i] == '--max-files':
      max_files = int(sys.argv[i+1])

  # Split the dataset
  print('Splitting Dataset...')
  split_image_dataset(dataset_path, split_path, max_files=max_files, train=train, val=val)
  print('Splitting Complete!')
  sys.exit(0)
