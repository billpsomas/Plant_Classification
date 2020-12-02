from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from create_generators import create_generators
from plantnet import PlantNet
from calculate_test_accuracy import calculate_test_acccuracy
from schedule import schedule

import tensorflow as tf
import numpy as np
import os
import datetime

K.clear_session()

# Define the train and test dataset directories
dataset_dir = "D:/advent/data/100ClassesPlantCLEF2017Train1EOL/augmented_data/"
test_dataset_dir = "D:/advent/data/100ClassesPlantCLEF2017TestEOL/data/"

plantclef_classes_list = [x for x in os.listdir(dataset_dir)]
plantclef_classes = len(plantclef_classes_list)

train_generator, val_generator, test_generator, height, width, batch_size = create_generators(dataset_dir, test_dataset_dir, 224, 224, 64)

# Define the model
model, name = PlantNet(height, width, plantclef_classes)

model.summary()
    
lr_scheduler = LearningRateScheduler(schedule)

model.compile(optimizer='Nadam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy','top_k_categorical_accuracy'])

# Set the number of epochs for training
num_epochs = 20

model_name = name
check_dir = "D:/advent/checkpoints/" + model_name + "/" + str(plantclef_classes) + "_classes/"
num_train_images = train_generator.n
num_val_images= val_generator.n
num_test_images= test_generator.n

try:
  os.makedirs(check_dir)
except:
  FileExistsError

step_size_train= num_train_images//train_generator.batch_size
step_size_valid= num_val_images//val_generator.batch_size
step_size_test= num_test_images//test_generator.batch_size

filepath= check_dir + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"

checkpointer = ModelCheckpoint(filepath=filepath,
                               verbose=1,
                               save_best_only=True)

csv_logger = CSVLogger(check_dir + 'hist_' + str(num_epochs) +'_epochs'+model_name+'.log')

history = model.fit_generator(train_generator, epochs=num_epochs, 
                              steps_per_epoch=num_train_images // batch_size,
                              callbacks = [csv_logger, checkpointer, lr_scheduler],
                              validation_data = val_generator,
                              verbose=1,
                              validation_steps = num_val_images // batch_size)

model = tf.keras.models.load_model("D:/advent/checkpoints/NASNetMobile/100_classes/NASNetMobile20191217-164128.h5")
accuracy = calculate_test_accuracy(train_generator, test_generator, model)
