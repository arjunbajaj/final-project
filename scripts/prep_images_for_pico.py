# |-------------------------------------|
# |       [CM3070] FINAL PROJECT        |
# |    CLASSIFYING SATELLITE IMAGERY    |
# |    BSc CS - UNIVERSITY OF LONDON    |
# |             ARJUN BAJAJ             |
# |-------------------------------------|

# Convert Test Images for Pico
# -----------------------------
# The Tensorflow Lite CNN model I have trained requires images
# to be in float32 format. I tried to convert the images directly
# on the Pico, but due to memory issues, it was unfeasible.
# Converting them beforehand and sending a binary of floats
# to the Pico works better.

import os, shutil

# Disable Tensorflow from logging unnecessary information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

# Delete and create a new folder to store the binary converted images
shutil.rmtree("../datasets/pico-test-images", ignore_errors=True)
os.makedirs("../datasets/pico-test-images", exist_ok=True)

# Load the EuroSAT test images
images = tf.keras.utils.image_dataset_from_directory(
  '../datasets/eurosat/test',
  image_size=(64, 64),
  shuffle=True
).unbatch().as_numpy_iterator()

# Counter to keep track of the number of images converted
index = 0

print("\n\n\nStarting to process images...")

for image, label in images:
  # Encode the label in the filename
  filename = f"../datasets/pico-test-images/{index:04d}_{label}.bin"

  # Print status
  print(f"Writing {index:04d}_{label}.bin", end="\r")

  # Convert the image to a binary file of floats,
  # as TensorFlow expects the input images to be in this format
  image = image.astype(np.float32).tobytes()

  # Write the binary to disk
  with open(filename, "wb") as f:
    f.write(image)

  index += 1
  if index >= 1000: break

print(f"Completed writing {index} files.")
