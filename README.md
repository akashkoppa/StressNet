# StressNet
This repository contains codes and data required to generate deep learning formulations of transpiration stress using Tensorflow. 

Author: Akash Koppa 

Contact: `akash.koppa@ugent.be`

## System Requirements

Operating System (OS): Linux, MacOS, Windows.

Software: conda (version > 4.0) - Anaconda, Miniconda or anyother virtual environment.

Optional Software: CUDA to use GPU support (Please consult the relevant instructions from NVIDIA for specific operating systems and graphics cards). 

**StressNet has been tested on the following specifications**:
```
Operating System:    Elementary OS 6.1 (Jólnir) 
                     (Based on Ubuntu 20.04.3 LTS)
Linux Kernel:        Linux Kernel 5.13.0-25-generic
Processor (CPU):     AMD Ryzen threadripper 3960x 24-core
Memory:              128gb
Graphics Card (GPU): NVIDIA GeForce RTX 2070 Super
NOTE: GPU is optional. 
```

## Installation Guide

The commans in this installation guide is based on the Linux-based OS. Similar commands can be used in MacOS (terminal) or Windows (Anaconda terminal). The following commands should be entered in the terminal.

1. Create a conda environment and activate it.
```
conda create --name stressnet
conda activate stressnet
```

2. Install the required packages and libraries
```
conda install pip
pip install -- upgrade tensorflow
pip install pandas
pip install tables
pip install matplotlib
```
Typical install time: ~30 minutes for all the packages.

NOTE: Resource for installing tensorflow including activating GPU support can be found in: https://www.tensorflow.org/install/

## Demo Guide
The files in the `demo` folder demonstrates the training of the stressnet for a subset of the data used in the research article (details in the usage guide). The folder consists of the following files:

1. `train_demo.py`: this is the main python script which consists of the code to train a deep neural net (stressnet) for the given input data.
2. `stressnet_functions.py`: contains functions to preprocess the input data into a format amenable to tensorflow, the Kling-Gupta efficiency (KGE) loss function, and the deep learning architecture of the stressnet. 
3. `input` folder: containing the input files required for training the StressNet.

NOTES:
1. Please use the `train_demo.py` script to test the code. The code is well-commented and the sequence of steps required to create the StressNet is logically designed.
2. Any changes to the hyperparameters or the input data configurations can be done in the `stressnet_functions.py` script. 

**Expected Outcome**: 1) A deep learning model based on the input data. 2) A graph showing the evolution of the training process in terms of the changes in the loss function (KGE in this case).

**Expected Run Time**: ~45 min. 

## Usage Guide
1. The `python` scripts used to generate the final StressNet formulations of transpiration stress for tall (`train_tall_vegetation.py`) and short (`train_short_vegetation.py`) vegetation (presented in the research article) alongwith the entire dataset is in the `stressnet` folder. The usage is similar to the demo file. 
2. The final trained StressNet formulations for both tall and short vegetation are present in `hybrid_paper/trained_stressnet` folder
3. The expected runtime for training both the tall and short vegetation is ~3 hours. 

### Reference

Koppa, A., Rains, D., Hulsman, P., & Miralles, D. (2021). A Deep Learning-Based Hybrid Model of Global Terrestrial Evaporation. Preprint. DOI: 10.21203/rs.3.rs-827869/v1.

Description: A version of StressNet for tall and short vegetation has been combined with a process-based model - Global Land Evaporation Amsterdam Model (GLEAM) - to create a hybrid model of global terrestrial evaporation. The resulting deep learning formulations and the hybrid model are detailed in the preprint. The manuscript is currently under review.

**NOTE**: All the data and codes required to reproduce the figures in the reference research article are available in the `hybrid_paper` folder. The data for the figures are available from https://doi.org/10.5281/zenodo.5886608



