# |-------------------------------------|
# |       [CM3070] FINAL PROJECT        |
# |    CLASSIFYING SATELLITE IMAGERY    |
# |    BSc CS - UNIVERSITY OF LONDON    |
# |             ARJUN BAJAJ             |
# |-------------------------------------|

# Script to run inference on the Pico
# ------------------------------------
# The Pico should be flashed with the build image.
# This script starts and waits for a Pico to be connected.
# It then sends each image to the Pico, when then runs inference
# prints the results. The results are parsed, collected, and finally
# written to a CSV file.

import time, sys, pathlib, os, serial, pathlib
import numpy as np

dataset_path = pathlib.Path("../datasets/pico-test-images").expanduser()
results_path = pathlib.Path("../artifacts/pico-inference-results.csv").expanduser()
results_path.exists() and results_path.unlink()

# append a line to a file
def append_line(values):
  with open(results_path, 'a') as f:
    f.write(','.join(str(v) for v in values) + '\n')

append_line(['path', 'true', 'pred', 'time'])

# A function to generate the test images
def test_images_generator():
  test_images = np.sort(os.listdir(dataset_path))

  max_images = len(test_images)
  print(f'Found {len(test_images)} images in {dataset_path}, using {max_images}.')

  test_images = test_images[0:max_images]

  for image_path in test_images:
    with open(dataset_path / image_path, "rb") as image_file:
      label = image_path.split(".")[0].split("_")[1]
      image_bytes = image_file.read()
      yield image_path, label, image_bytes

  return None


# Send the image to the pico, and parse its output
def run_inference_on_one_image(pico, image_bytes):
  pico.write(image_bytes)

  values = []
  category = -1
  time_taken = -1

  while True:
    s = pico.readline()
    s = s.decode('utf-8').split("\r\n")[0].strip()
    if debug: print('DEBUG FROM PICO: ', s)

    if "RESULT" in s:
      category = int(s.split(":")[1])

    if "TIME" in s:
      time_taken = int(s.split(":")[1].split(" ms")[0])

    if "WAIT" in s:
      print("Pico is waiting after processing 25 images...")

    if "VALUES" in s:
      values_str = s.split(":")[1].strip()
      values_list = values_str.split(",")
      values = [float(v) for v in values_list if v != ""]

    if "READY" in s:
      break

  return category, time_taken, values


# Send images to pico, and collect the results
def run_inference(pico):
  print('Sending images...')
  counter = 0
  for image_path, label, image_bytes in test_images_generator():
    counter += 1
    category, time_taken, _values = run_inference_on_one_image(pico, image_bytes)
    print(f"{counter} - {image_path}\tTrue: {label}\tPred: {category}\tTime: {time_taken}")
    append_line([image_path, label, category, time_taken])


# This script is built to run from the command line
if __name__ == '__main__':
  start_time = time.time()

  debug = False

  if len(sys.argv) > 1 and sys.argv[1] == "--debug":
    print("Debug mode enabled")
    debug = True

  pico_device = None
  msg_already_printed = False

  # Wait for a Pico device to be connected.
  # On Mac, the device is connected to /dev/tty.usbmodem<...>
  # and on Linux, the device is connected to /dev/ttyACM<...>
  while not pico_device:
    for device in os.listdir("/dev"):
      if device.startswith("tty.usbmodem") or device.startswith("ttyACM"):
        pico_device = device
      else:
        if not msg_already_printed:
          print("Please connect your Pico to the computer now.")
          msg_already_printed = True

  print(f"Pico is connected at: /dev/{pico_device}")
  print("Preparing to run Inference on Pico")

  # Connect to the Pico over serial
  pico = serial.Serial(f"/dev/{pico_device}")

  # Read output from the Pico until it sends a "READY" message
  while True:
    s = pico.readline()
    s = s.decode('utf-8')
    if debug: print('DEBUG FROM PICO: ', s)
    if "READY" in s:
      print("Pico is ready to run inference")
      break

  # Start sending data when the Pico is ready!
  run_inference(pico)

  # Close the serial connection
  pico.close()

  print(f"Inference on Pico completed in {time.time() - start_time :d} seconds")
  print("Results are saved to ../artifacts/pico-inference-results.csv")
  print("You can safely disconnect the Pico from the computer now.")
