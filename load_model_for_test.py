import tensorflow as tf

from utils.create_generators import create_generators
from utils.calculate_test_accuracy import calculate_test_acccuracy

dataset_dir = "D:/advent/data/100ClassesPlantCLEF2017Train1EOL/augmented_data/"
test_dataset_dir = "D:/advent/data/100ClassesPlantCLEF2017TestEOL/data/"
model_dir = "D:/advent/checkpoints/NASNetMobile/100_classes/NASNetMobile20191217-164128.h5"

model = tf.keras.models.load_model(model_dir)

train_generator, val_generator, test_generator, height, width, batch_size = create_generators(dataset_dir, test_dataset_dir, 224, 224, 64)

accuracy = calculate_test_accuracy(train_generator, test_generator, model)