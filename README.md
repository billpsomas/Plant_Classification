<h1 align="center">
 Plant Classification
</h2>
<p align="center">

## Overview
This repository contains code written explicitly in [**Python 3**](https://www.python.org/) for plant classification. [Plant classification](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/plantmaterials/technical/toolsdata/plant/?cid=stelprdb1043051) or [plant taxonomy](https://en.wikipedia.org/wiki/Plant_taxonomy) is the science that finds, identifies, describes, classifies and names plants by placing them in a hierarchical structure, where each level is being given a name (e.g., kingdom, division (phylum), class, order, family, genus, species). 

We use [**Tensorflow 2.0**](https://www.tensorflow.org/) alongside with other libraries for the implementation. We also use the [**PlantCLEF2017**](https://www.imageclef.org/lifeclef/2017/plant) competition dataset, which consists of the "trusted" training set based on the online collaborative [**Encyclopedia Of Life (EoL)**](https://eol.org/), the "noisy" training set built through web crawlers (Google and Bing image search results) and the test dataset, which is a large sample of the raw query images submitted by the users of the mobile application [**Pl@ntNet**](https://play.google.com/store/apps/details?id=org.plantnet).

The documentation that follows is meant to be as detailed as possible, so that anyone can follow, no matter his level.

## Install the necessary libraries
The first step in order to run this code is to download the necessary libraries. Python has a tremendous amount of libraries and this is one of its strongest advantages. Python comes with a package installer and manager called [PIP](https://pypi.org/project/pip/), which is by default installed from version 3.4 onwards. In order to install the libraries, run the following command on your terminal:
```
pip install -r requirements.txt
```

## Download datasets
No matter your operating system, whether you are using Windows, Linux, etc., open your terminal and write the following commands in order to download the datasets. Be careful to navigate through your directories in order to download the datasets in a directory you will later remember. We suggest creating a new directory for this named `data` using the commands:
```
$ cd plant_classification
$ mkdir data
$ cd data
```
You must now have a `plant_classification/data` directory inside which you will download the datasets.
- Download the EoL training dataset:
```
$ wget http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2017/TrainPackages/PlantCLEF2017Train1EOL.tar.gz
```
- Download the Pl@ntNet test dataset:
```
$ wget http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2017/TestPackage/PlantCLEF2017Test.tar.gz
```

## Extract files
For **Linux** machines:
- Extract the EoL training set:
```
$ tar -xfv PlantCLEF2017Train1EOL.tar.gz
```
- Extract the Pl@ntNet test set:
```
$ tar -xfv PlantCLEF2017Test.tar.gz
```
For **Windows** PCs:

The corresponding way using the terminal (PowerShell) needs a function like the following one that expands the 7Zip functionality. Run this on your terminal: 
```
function Expand-Tar($tarFile, $dest) {

    if (-not (Get-Command Expand-7Zip -ErrorAction Ignore)) {
        Install-Package -Scope CurrentUser -Force 7Zip4PowerShell > $null
    }

    Expand-7Zip $tarFile $dest
}
```
And then:
- Extract the EoL training set:
```
Expand-Tar PlantCLEF2017Train1EOL.tar.gz PlantCLEF2017Train1EOL
Expand-Tar PlantCLEF2017Train1EOL.tar PlantCLEF2017Train1EOL
```
- Extract the Pl@ntNet test set:
```
Expand-Tar PlantCLEF2017Test.tar.gz PlantCLEF2017Test
Expand-Tar PlantCLEF2017Test.tar PlantCLEF2017Test
```
Of course, once you are using Windows, there is also the naive way of using a software like [7zip](https://www.7-zip.org
) to unzip those datasets in the respective directories.

## Issues
There are several issues concering these datasets:
1. The distribution of samples of every class in training set is not uniform. There is the so called long-tail problem.
2. There is no explicit split between training and validation.
3. The test set consists of raw query images, the classes of whom are not equal to the number of classes of the training set. With other words, there are some classes that do not have any sample on test set.

## Solution
We use the Tensorflow 2.0 [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) in order to process the images. Moreover, we keep the extracted `data/PlantCLEF2017Train1EOL` training directory with the 10000 classes, but create a new directory named `data/PlantCLEF2017StructuredTest`, which is aimed to be more structured than the `data/PlantCLEF2017Test` (concerning the way the images are organized in folders) and to include test images for all the classes. We follow the formulation of the training directory and create folders named after the classes to which they correspond. We fill those folders with images of the corresponding classes. To do so, run:
```
python create_structured_test_set.py
``` 

We notice that the number of classes for which we have available test data is really small, as it is about the 1/5 of the classes. To confront this, we find the classes that are missing test data and add data coming from the training set to them. Moreover, we add 10% of images of the corresponding training class to each test class that is missing data. In case the training class has less than 10 images, then we add just 1 image. To do so, run:
```
python fill_structured_test_set.py
``` 