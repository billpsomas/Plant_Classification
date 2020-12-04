'''THIS SCRIPT IS CREATED BECAUSE OF THE FACT THAT THE TEST DATASET HAS A LOT OF MISSING CLASSES. 
    WE FIND THESE CLASSES AND REMOVE 10% OF THEIR RESPECRIVE TRAINING CLASSES FILES FROM TRAINING DATASET 
    TO TEST DATASET'''

import os
import shutil
import random

from pathlib import Path

# Get the current working directory
dir = os.getcwd()

train_dataset_dir = Path(dir + "/data/PlantCLEF2017Train1EOL/data/")
test_dataset_dir = Path(dir + "/data/PlantCLEF2017StructuredTest/data/")

train_classes = [x for x in os.listdir(train_dataset_dir)]
test_classes = [x for x in os.listdir(test_dataset_dir)]

print("There is a problem because the number of train classes is {}, while the number of test classes is {}".format(len(train_classes), len(test_classes)))

missing_classes = [x for x in train_classes if x not in test_classes]

print("There are {} missing classes from the test dataset".format(len(missing_classes)))

for i in range (len(missing_classes)):
    missing_class_files = [x for x in os.listdir(train_dataset_dir + missing_classes[i]) if x.endswith("jpg")]
    num_of_files_to_be_removed = max(1, int(len(missing_class_files) * 0.1))
    files_to_be_removed = random.sample(missing_class_files, num_of_files_to_be_removed)
    try:
        os.makedirs(test_dataset_dir + missing_classes[i])
    except:
        FileExistsError
    for j in range(len(files_to_be_removed)):
        shutil.move(train_dataset_dir + missing_classes[i] + "/" + files_to_be_removed[j], test_dataset_dir + missing_classes[i] + "/" + files_to_be_removed[j])
