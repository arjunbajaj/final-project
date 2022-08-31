# |-------------------------------------|
# |       [CM3070] FINAL PROJECT        |
# |    CLASSIFYING SATELLITE IMAGERY    |
# |    BSc CS - UNIVERSITY OF LONDON    |
# |             ARJUN BAJAJ             |
# |-------------------------------------|

# core.py
# -----------------------
# This file implements helper functions to:
#     1. Load the EuroSAT and UC Merced datasets.
#     2. Render the accuracy and loss charts and print the training duration.
#     3. Compile and Train the model.


# Disable TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt


def load_image_dataset(path, image_size=64, no_cache=False, batch_size=64):
  # Load the image dataset from directory
  dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(image_size, image_size),
    batch_size=batch_size,
    shuffle=True,
    seed=1701
  )

  if no_cache:
    return dataset
  else:
    # Prefetch and Cache the dataset
    return dataset.prefetch(tf.data.AUTOTUNE).cache()


# Load the EuroSAT dataset
def load_eurosat(image_size=64, **kwargs):
  train = load_image_dataset('../datasets/eurosat/train', image_size=image_size, **kwargs)
  val = load_image_dataset('../datasets/eurosat/val', image_size=image_size, **kwargs)
  test = load_image_dataset('../datasets/eurosat/test', image_size=image_size, **kwargs)
  return (train, val, test)

# Load the UC Merced dataset
def load_uc_merced(image_size=64, **kwargs):
  train = load_image_dataset('../datasets/uc_merced/train', image_size, **kwargs)
  val = load_image_dataset('../datasets/uc_merced/val', image_size, **kwargs)
  test = load_image_dataset('../datasets/uc_merced/test', image_size, **kwargs)
  return (train, val, test)


# Render the Accuracy Chart from the TensorFlow model.fit() history object.
def accuracy_chart(history):
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]
  epochs = range(1, len(accuracy) + 1)
  plt.figure()
  plt.plot(epochs, accuracy, "g", label="Training Accuracy")
  plt.plot(epochs, val_accuracy, "b", label="Validation Accuracy")
  plt.title("Training and Validation Accuracy")
  plt.legend()
  plt.show()


# Render the Loss Chart from the TensorFlow model.fit() history object.
def loss_chart(history):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  epochs = range(1, len(loss) + 1)
  plt.figure()
  plt.plot(epochs, loss, "g", label="Training Loss")
  plt.plot(epochs, val_loss, "b", label="Validation Loss")
  plt.title("Training and Validation Loss")
  plt.legend()
  plt.show()


# Print the amount of time it takes to train the model
def print_timing(training_time, epochs, history):
  mins = int(training_time // 60)
  secs = int(training_time % 60)
  actual_epochs = len(history.history['loss'])

  if actual_epochs != epochs:
    print(f'\nTrained for {actual_epochs} epochs (stopped early) in {mins}m{secs}s.')
  else:
    print(f'\nTrained for {epochs} epochs in {mins}m{secs}s.')


# A helper function to compile the model using the provided learning rate,
# as the optimizer, loss, and metric are the same for the entire project.
def compile_model(model, learning_rate=0.001):
  # allow setting a custom learning rate for the optimizer
  optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

  # compile the model using the loss function and optimizer
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model


# A helper function to run model.fit() with the appropriate configuration
# and render charts and results.
def train_model(train_ds, val_ds, model, model_filename, epochs=10,
  early_stopping=False, patience=5, tensorboard=False
):

  # save the best model to disk
  callbacks = [
    keras.callbacks.ModelCheckpoint(
      filepath=f'../models/{model_filename}.keras',
      save_best_only=True,
      monitor='val_loss',
    )
  ]

  # Log model training to visualize in TensorBoard
  if tensorboard:
    callbacks.append(keras.callbacks.TensorBoard(log_dir=f"../models/logs/{model_filename}_logs"))

  # Stop training early if validation accuracy stangates for more than 5/`patience` epochs
  if early_stopping:
    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience))

  # start timing
  training_time_start = time.time()

  # train the model
  history = model.fit(
    train_ds.cache(),
    epochs=epochs,
    validation_data=val_ds.cache(),
    callbacks=callbacks
  )

  # time for training
  training_time = time.time() - training_time_start

  # render the accuracy and loss charts and print training duration
  accuracy_chart(history)
  loss_chart(history)

  val_model = keras.models.load_model(f'../models/{model_filename}.keras')
  [_loss, accuracy] = val_model.evaluate(val_ds.cache())
  print_timing(training_time, epochs, history)
  print(f'Highest Validation Accuracy: {accuracy * 100 :.2f}%')

  return history


# A helper function to compile and train the model using the provided datasets.
def compile_and_train_model(
  model, filename, dataset,
  epochs=10, learning_rate=0.001, tensorboard=False,
  early_stopping=True, patience=5
):
  # load the data
  train, val, _test = dataset

  # compile model
  model = compile_model(model, learning_rate=learning_rate)

  # train model
  train_model(
    train, val, model, model_filename=filename, epochs=epochs,
    early_stopping=early_stopping, patience=patience, tensorboard=tensorboard
  )
