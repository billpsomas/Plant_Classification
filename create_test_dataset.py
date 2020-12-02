import pandas as pd
import os
import shutil

# Read the CSV having the info about the test images
df = pd.read_csv("/home/athena/PythonEnvs/tf2env/data/PlantCLEF2017Test/PlantCLEF2017OnlyTest.csv", delimiter=";", error_bad_lines=False)

# Create a list with all the unique class names of the test set
classes = df["ClassId"].unique().tolist()

for i in range(len(classes)):
    os.makedirs("/home/athena/PythonEnvs/tf2env/data/PlantCLEF2017TestEOL/data/" + str(classes[i]))

for j in range(len(df.index)):
    try:
        shutil.move("/home/athena/PythonEnvs/tf2env/data/PlantCLEF2017Test/data/" + str(df['MediaId'][j]) + ".jpg",
                    "/home/athena/PythonEnvs/tf2env/data/PlantCLEF2017TestEOL/data/" + str(df['ClassId'][j]) + "/" + str(df['MediaId'][j]) + ".jpg")
    except:
        FileNotFoundError