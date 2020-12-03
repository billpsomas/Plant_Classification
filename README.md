<h1 align="center">
 Plant Classification
</h2>
<p align="center">

## Overview
This repository contains code written explicitly in [**Python 3**](https://www.python.org/) for plant classification. [Plant classification](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/plantmaterials/technical/toolsdata/plant/?cid=stelprdb1043051) or [plant taxonomy](https://en.wikipedia.org/wiki/Plant_taxonomy) is the science that finds, identifies, describes, classifies and names plants by placing them in a hierarchical structure, where each level is being given a name (e.g., kingdom, division (phylum), class, order, family, genus, species). 

We use [**Tensorflow 2.0**](https://www.tensorflow.org/) alongside with other libraries for the implementation. We also use the [**PlantCLEF2017**](https://www.imageclef.org/lifeclef/2017/plant) competition dataset, which consists of the "trusted" training set based on the online collaborative [**Encyclopedia Of Life (EoL)**](https://eol.org/), the "noisy" training set built through web crawlers (Google and Bing image search results) and the test dataset, which is a large sample of the raw query images submitted by the users of the mobile application [**Pl@ntNet**](https://play.google.com/store/apps/details?id=org.plantnet).

## Install the necessary libraries
```
pip install -r requirements.txt
```

## Download datasets
No matter your operating system, whether you are using Windows, Linux, etc., open your terminal and write the following commands in order to download the datasets. Be careful to navigate through your directories in order to download the datasets in a directory you will later remember. We suggest creating a new directory for this named "data" using the commands:
```
$ cd plant_classification
$ mkdir data
$ cd data
```

- Download the EoL training dataset:
```
$ wget http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2017/TrainPackages/PlantCLEF2017Train1EOL.tar.gz
```
- Download the Pl@ntNet test dataset:
```
$ wget http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2017/TestPackage/PlantCLEF2017Test.tar.gz
```

## Extract files
- Extract the EoL training set:
```
$ tar -xfv PlantCLEF2017Train1EOL.tar.gz
```
- Extract the Pl@ntNet test set:
```
$ tar -xfv PlantCLEF2017Test.tar.gz
```

## Issues
There are several issues concering these datasets:
1. The distribution of samples of every class is not uniform. 