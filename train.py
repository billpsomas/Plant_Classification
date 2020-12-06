# Importing necessary modules of Tensorflow
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

#from create_generators import create_generators

# Importing the models
from models.plantnet import PlantNet
from models.resnet50v2 import ResNet50V2
from models.nasnetmobile import NASNetMobile
from models.inceptionv3 import InceptionV3
from models.inceptionresnetv2 import InceptionResNetV2

from utils.calculate_test_accuracy import calculate_test_acccuracy
from utils.schedule import schedule
from utils.plot_figures import plot_figures

import tensorflow as tf
import numpy as np
import os
import datetime
import argparse

from pathlib import Path

parser = argparse.ArgumentParser(description= 'Plant Classification using Tensorflow 2.0')

# Set the number of epochs for training
parser.add_argument('--epochs', default = 20, 
                                type = int, 
                                dest = 'num_epochs',
                                help = 'Number of training epochs')
# Set the model 
parser.add_argument('--model', default = 'PlantNet',
                               type = str,
                               dest = 'model',
                               help = 'Model to be used')
# Set the batch size
parser.add_argument('--batch_size', default = 64,
                                    type =  int,
                                    dest = 'batch_size',
                                    help = 'Batch size to be used')
# Set the height of input images
parser.add_argument('--height', default =224,
                                type = int,
                                dest = 'height',
                                help = 'Height of input images, input shape = (height, width, channels)')
# Set the width of input images
parser.add_argument('--width', default =224,
                                type = int,
                                dest = 'width',
                                help = 'Width of input images, input shape = (height, width, channels)')

args = parser.parse_args()

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
train_generator, val_generator, test_generator, height, width, batch_size = create_generators(dataset_dir, test_dataset_dir, args.height, args.width, args.batch_size)

# Initialize model
if args.model == 'PlantNet':
  model, name = PlantNet(height, width, plantclef_classes)
elif args.model == 'ResNet50V2':
  model, name = ResNet50V2(height, width, plantclef_classes)
elif args.model == 'NASNetMobile':
  model, name = NASNetMobile(height, width, plantclef_classes)
elif args.model == 'InceptionV3':
  model, name = InceptionV3(height, width, plantclef_classes)
elif args.model == 'InceptionResNetV2':
  model, name = InceptionResNetV2(height, width, plantclef_classes)

# Print a model summary
model.summary()

# Define scheduler
lr_scheduler = LearningRateScheduler(schedule)

# Compile to model
# Define optimizer, loss and metrics 
model.compile(optimizer='Nadam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy','top_k_categorical_accuracy'])

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
csv_logger = CSVLogger(Path(str(check_dir) + "/" + 'hist_' + str(args.num_epochs) + '_epochs' + model_name + '.log'))

# Train the model
history = model.fit_generator(train_generator, epochs=args.num_epochs, 
                              steps_per_epoch=num_train_images // batch_size,
                              callbacks = [csv_logger, checkpointer, lr_scheduler],
                              validation_data = val_generator,
                              verbose=1,
                              validation_steps = num_val_images // batch_size)

# Load the model after training in order to calculate test accuracy
model = tf.keras.models.load_model(filepath)
accuracy = calculate_test_accuracy(train_generator, test_generator, model)
