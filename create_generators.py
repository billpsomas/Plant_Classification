import tensorflow as tf
import numpy as np
import os

def create_generators(model_train_dataset_dir, model_test_dataset_dir, height, width, batch_size):

    datagen =  tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split= 0.2
        )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        )

    train_generator = datagen.flow_from_directory(directory= model_train_dataset_dir,
                                                shuffle= True,
                                                target_size=(height, width), 
                                                batch_size=batch_size,
                                                subset='training'
                                                )

    val_generator = datagen.flow_from_directory(directory= model_train_dataset_dir,
                                                shuffle= True,
                                                target_size=(height, width), 
                                                batch_size=batch_size,
                                                subset='validation'
                                                )

    test_generator = test_datagen.flow_from_directory(directory= model_test_dataset_dir,
                                                shuffle= False,
                                                target_size=(height, width),
                                                batch_size=1,
                                                class_mode=None
                                                )
    
    return train_generator, val_generator, test_generator, height, width, batch_size