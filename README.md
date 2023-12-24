# CNN Model

**Introduction**

We present a CNN designed to predict the fluorescence levels on a scale of 1-4 of a DNA sequences.
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


# DNA Sequence Prediction Readme

## Overview ##


This Python script enables the prediction of numerical gene expression values for given DNA sequences using pre-trained deep learning models. The models: ADM, AMM, and MBO, predict mean flourescence based on the dna sequence of synthetic upstream regulatory region. This readme provides information on how to use the script, prerequisites for execution, and details about the models and input data.

**Prerequisites**


Before using the script, ensure that the following dependencies are installed:

* argparse: A module for parsing command-line arguments.<br>
* pandas: A powerful data manipulation library.<br>
* keras: A high-level neural networks API.<br>
* numpy: A fundamental package for scientific computing with Python.<br>
* scipy: A library for scientific computing and statistical routines.<br>

**Usage**


Execute the script from the command line using the following syntax:<br>

```python predict.py model_name output_file input_file```<br>

* model_name: Specify the model to be used for predictions (ADM, AMM, or default MBO).<br>
* output_file: The name of the file where predictions will be saved.<br>
* input_file: The name of the file containing input DNA sequences.<br>

**DNA Sequence Encoding**


The script employs a one-hot encoding scheme to represent DNA sequences. Each nucleotide is mapped to a binary vector. The mapping is as follows:

* "A": [1, 0, 0, 0]<br>
* "C": [0, 1, 0, 0]<br>
* "G": [0, 0, 1, 0]<br>
* "T": [0, 0, 0, 1]<br>
* "K": [0, 0, 0.5, 0.5]<br>
* "M": [0.5, 0.5, 0, 0]<br>
* "N": [0.25, 0.25, 0.25, 0.25]<br>

**Loading Models**


The script loads pre-trained models based on the specified model_name. Currently available models are:

* ADM (All Data Model) - Trained on 20,000 sequences with the highest number of reads from the expreiment<br>
* AMM (All Motif Model)- Trained on 2,435 sequences with 22 barcodes each <br>
* MBO (Mixed Bases Only Model)- Trained on 2,098 sequences with 22 barcodes and at least one mixed base (K/M) <br>

**Input File Format**


The input file should contain DNA sequences of synthetic upstream regulatory region, with each sequence on a new line. The sequence should be 101 bases in length for the models. The script reads these sequences from the input file.

**Output**


The script outputs predictions for each input sequence to the specified output_file. Each prediction is written to a new line.

**Example**


```python predict.py ADM predictions.txt input_sequences.txt```<br>

This command runs the script using the ADM model, with input sequences from the file input_sequences.txt, and saves the predictions to the file predictions.txt.



  
