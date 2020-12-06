# Importing necessary modules of Tensorflow
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from create_generators import create_generators

# Importing the models
from models.plantnet import PlantNet
from models.resnet50v2 import ResNet50V2
from models.nasnetmobile import NASNetmobile
from models.inceptionv3 import InceptionV3
from models.inceptionresnetv2 import InceptionResNetV2

from calculate_test_accuracy import calculate_test_acccuracy
from schedule import schedule

import tensorflow as tf
import numpy as np
import os
import datetime

from pathlib import Path

import pdb
pdb.set_trace()

# Get the current working directory
dir = os.getcwd()

# Clear memory
K.clear_session()

# Define the train and test dataset directories
dataset_dir = Path(dir + "/data/PlantCLEF2017Train1EOL/augmented_data/")
test_dataset_dir = Path(dir + "/data/PlantCLEF2017StructuredTest/data/")

plantclef_classes_list = [x for x in os.listdir(dataset_dir)]
plantclef_classes = len(plantclef_classes_list)

# Create the generators
train_generator, val_generator, test_generator, height, width, batch_size = create_generators(dataset_dir, test_dataset_dir, 224, 224, 64)

# Define the model
model, name = PlantNet(height, width, plantclef_classes)

# Print a model summary
model.summary()

# Define scheduler
lr_scheduler = LearningRateScheduler(schedule)

# Compile to model
# Define optimizer, loss and metrics 
model.compile(optimizer='Nadam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy','top_k_categorical_accuracy'])

# Set the number of epochs for training
num_epochs = 20

# Set up a checkpoint directory
model_name = name
check_dir = Path(str(dir) + "/" + "checkpoints/" + model_name + "/" + str(plantclef_classes) + "_classes/")

# Calculate the training, validation and test images
num_train_images = train_generator.n
num_val_images= val_generator.n
num_test_images= test_generator.n

# Make the checkpoint directory
try:
  os.makedirs(check_dir)
except:
  FileExistsError

# Calculate the step sizes 
step_size_train= num_train_images//train_generator.batch_size
step_size_valid= num_val_images//val_generator.batch_size
step_size_test= num_test_images//test_generator.batch_size

# Set up the file in which the model will be saved after training
filepath= Path(str(check_dir) + "/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5")

# Initialize the checkpointer
checkpointer = ModelCheckpoint(filepath=filepath,
                               verbose=1,
                               save_best_only=True)

# Initialize the CSVLogger
csv_logger = CSVLogger(Path(str(check_dir) + "/" + 'hist_' + str(num_epochs) + '_epochs' + model_name + '.log'))

# Train the model
history = model.fit_generator(train_generator, epochs=num_epochs, 
                              steps_per_epoch=num_train_images // batch_size,
                              callbacks = [csv_logger, checkpointer, lr_scheduler],
                              validation_data = val_generator,
                              verbose=1,
                              validation_steps = num_val_images // batch_size)

# Load the model after training in order to calculate test accuracy
model = tf.keras.models.load_model(filepath)
accuracy = calculate_test_accuracy(train_generator, test_generator, model)
