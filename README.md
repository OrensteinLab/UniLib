# CNN Model

**Introduction

We present a CNN desinged to predict the fluorescence levels on a scale of 1-4 of a DNA sequences.
As a part of this study we generated dataset comprising of around 150k DNA sequences, each sequence is 116 long with the first 15 nt being a barcode,
it is possible to ignore the first 15 nt (we discovered it doesnt change the results).

This repository contains 4 different models:

  * bins - this model's output is a 1x4 probabilty vector 
  
  * bins_sample_weights - similar to "bins" model but is takes into consideration the amount of total reads each sequence has

  * meanFL - the output of this model is a single scalar value called meanFL calculated by the formula : p(bin1)*607 + p(bin2)*1364 + p(bin3)*2596 + p(bin4)*7541       (p is the probabilty of the sequence being in the x bin)

  * meanFL_sample_weights - similar to "meanFL" model but is takes into consideration the amount of total reads each sequence has 

**Getting started**

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

**Prerequisites**

  * pandas
  
  * numpy
  
  * tensorflow
  
  * keras
  
  * scipy
  
  * sklearn
  
  **How it works**
  
  **Training the model:**
  
  * In order to train the model on the same data:
  
  first you need to choose a model from the 8 models we compared then you simply have to replace this line:
  
  ![image](https://user-images.githubusercontent.com/101515707/177045823-2edb66a4-3a76-48df-b430-8dbaf21a93c7.png)
    
    with a line that loads the data to your machine.
    
  * To train the model on another set of data you need 
  
  A) To choose a model and make sure your data contains the relevant features e.g bin readings or meanFL
  
  B) Replace this line:
  
  ![image](https://user-images.githubusercontent.com/101515707/177045823-2edb66a4-3a76-48df-b430-8dbaf21a93c7.png)
  
  with a line that loads your own data.
  
  C) Make sure the labels are loaded correctly into the following veriables:
  
  ![image](https://user-images.githubusercontent.com/101515707/177046169-451b154b-f47a-45c6-b512-eb612a97f395.png)
  
  **Getting predictions:**
  
  A) Transform the sequences into a one hot matrix representation of the string using the oneHotDeg function:
  ![image](https://user-images.githubusercontent.com/101515707/177046458-7cfd1ac1-04b9-4642-8ad3-d47767c9e2a4.png)

  B) Use model.predict(your_matrix_goes_here)
  

  
# Desert Sequence Generator

**Introduction** 

Here I developed a MATLAB code that generated a sequence devoid of any known yeast motifs (downloaded from: YeTFaSCo Version: 1.02), we termed it the desert sequence.

This sequence is the chassis for the synthetic upstream regulatory sequence OL, as well as the LexA bacterial regulatory sequence OL.

Sequence length can be adjustable, the code can be applied on number of shorter sequences and then on the combined sequences, to make a longer sequence (which 

also needs to get checked as desert), I found this makes overall shorter runtimes.


**How it works?**

To generate a desert sequence (devoid from any known yeast motifs, listed in YeastMotifs.xlsx file), start by converting 

the motifs into "regular expression" format, by running the DesertURSGenerator.m code in MATLAB.

This code will generate a regexps_YeastMotifs.xlsx output with all the motifs, in a regular expression description for each motif.

Then, the code randomizes a sequence in any desired length (default: 186bp) and replaces the yeast motifs found in the randomized sequence into a "desert".

The primary code uses the functions: CheckSeqValidity.m, CheckSequence.m, ManualRandSeq.m, and ReplaceMatchedSeq.m automatically.

It is important to save all files indluding functions and scripts in the same path.
 
Once finished, the desert sequence will appear on the MATLAB's command window but also saved in a "Results" .txt file.


# Using Transfer Learning - Pipeline

This repository contains a Python script for a deep learning model on DNA expression data that uses Transfer Learning. The script reads CSV files with expression data of many DNA sequences, preprocesses the data, trains a convolutional neural network (CNN) model with the transfer learning technique, and performs predictions on 2 validation sets with regulatory DNA sequences.

## Prerequisites

Before using the script, make sure you have the following libraries and tools installed:

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [SciPy](https://www.scipy.org/)
## Script Description

The script performs the following main tasks:

1. **Data Preprocessing**:
   - Reads a CSV file containing regulatory DNA sequences and expression measurements (mean fluorescence levels).
   - Converts the DNA sequences to one-hot encodings, enabling them to be used as input to the machine learning model.
   - Augments train sets and test sets with reverse complement sequences for training and prediction.

2. **Machine Learning Model**:
   - Defines a convolutional neural network (CNN) model for the task.
   - Compiles the model with appropriate loss and optimization functions.

3. **Model Training**:
   - Trains the CNN model on the preprocessed DNA sequences and their corresponding mean FL levels.
   - Uses transfer learning approach - trains model on 3 different datasets - from the largest and low-quality quality dataset to the smallest and high-quality dataset.

4. **Ensemble Model**:
   - Implements an ensemble method for model prediction. The script runs 100 models and averages their predictions to improve the accuracy and robustness of the model.

5. **Prediction**:
   - Uses the trained model to predict mean FL values for the validation sequences.
   - Calculates predictions based on the average predictions for the original and reverse complement sequences, improving prediction accuracy.

6. **Evaluation**:
   - Calculates Pearson correlation coefficients between the model predictions and the true labels for the 2 validation sets.
   - Evaluates model performance.

7. **Data Export**:
   - Saves the average predictions and true labels to CSV files for further analysis.

## Usage

To use this script, follow these steps:

1. **Install Required Libraries**:
   Ensure that you have installed all the required libraries and tools mentioned in the "Prerequisites" section.

2. **Data Preparation**:
   - Prepare your input data in CSV format, including DNA sequences and associated target values.
   - Name your input CSV files accordingly, or modify the script to specify the correct file paths.

3. **Configure the Script**:
   - Adjust the script to match your dataset and experimental setup. Update the file paths and parameters as needed.

4. **Run the Script**:
   - Execute the script using a Python interpreter.
   - The script will perform data preprocessing, model training, and evaluation.

  
  
  

  
