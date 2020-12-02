import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import lite

def convert2tflite(h5_file, tflite_file_name):
    converter = lite.TocoConverter.from_keras_model_file(h5_file)
    tflite_model = converter.convert()
    open(str(tflite_file_name), "wb").write(tflite_model)