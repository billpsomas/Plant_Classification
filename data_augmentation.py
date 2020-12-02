# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import os

input = "/home/athena/PythonEnvs/tf2env/data/100ClassesPlantCLEF2017TrainEOL/data/"
output = "/home/athena/PythonEnvs/tf2env/data/100ClassesPlantCLEF2017TrainEOL/augmented_data/"

try:
  os.makedirs(output)
except:
  FileExistsError

classes = os.listdir(input)

for i in range(len(classes)):
    for j in range(len(os.listdir(input + classes[i]))):
        if os.listdir(input + classes[i])[j].endswith(".jpg"):
            # Load the input images, convert them to numpy array, and then
            # reshape them to have an extra dimension
            print("[INFO] loading example image ", input + classes[i] + "/" + os.listdir(input + classes[i])[j])
            image = load_img(input + classes[i] + "/" + os.listdir(input + classes[i])[j])
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            
            # Construct the image generator for data augmentation then
            # initialize the total number of images generated thus far
            aug =  ImageDataGenerator(
                rescale=1.0/255.0,
                rotation_range=45,
                zoom_range= 0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.1,
                brightness_range=[0.5,1.5],
                horizontal_flip=True,
                fill_mode='nearest',
                )
            # Make the output folder in case it doesn't exist
            try:
                os.makedirs(output + classes[i])
            except:
                FileExistsError
            # Construct the actual Python generator
            print("[INFO] generating image...", )
            imageGen = aug.flow(image, batch_size=1, save_to_dir= output + classes[i] + "/", save_prefix="image", save_format="jpg")
            counter = 0
            desired_class_images = 2000
            class_images = len(os.listdir(input + classes[i]))
            images_to_be_created = desired_class_images - class_images
            total = round(desired_class_images/class_images)
            for image in imageGen:
                # Increment our counter
                counter += 1
                # If we have reached the speciied number of examples, break
                # from the loop
                if counter == total:
                    break