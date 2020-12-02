import tensorflow as tf
import numpy as np

def calculate_test_acccuracy(train_generator, test_generator, model):

    preds = model.predict_generator(test_generator)
    predicted_class_indices = np.argmax(preds, axis=1)
    true_class_indices = test_generator.classes

    inv_train_generator_indices  = {v: k for k, v in train_generator.class_indices.items()}
    inv_test_generator_indices = {v: k for k, v in test_generator.class_indices.items()}

    predicted_classes = [inv_train_generator_indices[c] for c in predicted_class_indices]
    true_classes = [inv_test_generator_indices[c] for c in true_class_indices]

    accuracy = np.sum(np.array(true_classes)==np.array(predicted_classes))/len(predicted_classes)
    print("The test set accuracy calculated is:", accuracy)

    return accuracy