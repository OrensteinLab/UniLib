# UniLib

# Table of Contents
1. [Introduction](#introduction)
2. [Setup environment](#setup-environment)
3. [Desert Sequence Generator](#desert-sequence-generator)
4. [DNA Sequence Prediction](#dna-sequence-prediction)
5. [Train model](#train-model)

# Introduction

In the UNILIB study, we leveraged data obtained from a Massive Parallel Reported Assay (MPRA) that systematically measured gene expression for an oligo library comprising approximately 150,000 Synthetic Upstream Regulatory Sequences (sURS) in yeast. This repository includes the collection of scripts used for machine learning and data analysis throughout the study. It also includes the datasets used in our investigation, as well as the trained machine learning models.

The repository also provides a script enabling predictions on new DNA sequence files using a user-selected machine learning model from the models used in the study. An additional script is available for training the CNN model used in the study on new DNA sequence and expression data.

# Setup environment


```
# Create a virtual conda environment named "Unilib" with Python 3.9.16
conda create -n Unilib python=3.9.16

# Activate the newly created virtual environment
conda activate Unilib

# Clone the UniLib project from the GitHub repository
git clone https://github.com/OrensteinLab/UniLib.git

# Change into the UniLib directory
cd UniLib

# Install the required dependencies listed in the requirements.txt file using pip
python -m pip install -r requirements.txt


```

  
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


# DNA Sequence Prediction

## Overview ##


This Python script enables the prediction of  gene expression values for given DNA sequences using pre-trained deep learning models. The models: ADM, AMM, and MBO, predict mean flourescence based on the dna sequence of synthetic upstream regulatory region. This readme provides information on how to use the script, prerequisites for execution, and details about the models and input data.

**Usage**

Execute the script from the command line using the following syntax:<br>

```

python predict.py model_name output_file input_file

```

* model_name: Specify the model to be used for predictions (ADM, AMM, MLAM, or default MBO).<br>
* output_file: The path of the file where predictions will be saved.<br>
* input_file: The path of the file containing input DNA sequences.<br>


**Sequence Encoding**
  

The script employs a one-hot encoding scheme to represent DNA sequences. Each nucleotide is mapped to a binary vector. The mapping is as follows:
<br>

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
* MLAM (ML Additive Model)- Trained on the MBO dataset and uses a motif count vector in addition to sequence during the training process <br>


**Input File Format**


The input file should contain DNA sequences of synthetic upstream regulatory region, with each sequence on a new line. The sequence should be 101 bases in length for the ADM,AMM and MBO models. For the MLAM model, the sequences can be up to the length 186bp. The script reads these sequences from the input file.


**Output**

The script outputs predictions for each input sequence to the specified output_file. Each prediction is written to a new line.

**Example**

<br>


```

python predict.py ADM predictions.txt input_sequences.txt

```

This command runs the script using the ADM model, with input sequences from the file input_sequences.txt, and saves the predictions to the file predictions.txt.

# Train model

## Overview ##

This Python script enables user to train the general ML model used in the UNILIB study on new DNA Sequence Expression data. The user provides and input file with the training sequences and expression labels. The trained model is saved as output


**Usage**


Execute the script from the command line using the following syntax:<br>

```

python train.py data_file model_name

```

* data_file: The path of the file containing DNA sequences and their mean FL values.<br>
* model_name: The name that the new trained model would be saved with


**Model Characteristics**

The model is based on a convolutional network (CNN) with the following hyperparameters:

* 1024 kernels with a filter size 6 and a relu activation function.
* A global max pooling layer
* A dense layer with 16 neurons and relu activation
* A final dense layer with 1 neuron and linear activation.
* The model use the MSE loss function and the ADAM optimizer.
* The batch size is 32 and the number of training epochs is 5.


**Input File Format**

The input file should contain DNA sequences of synthetic upstream regulatory region (sURS), and their mean FL values. Each sequence and its mean Fl value should be separated by a tab and should be on a separate line. The sequence should be 101 bases in length for the model. 

**Output**

The output of the program is the trained model saved in keras file format

**Example**


```

python train.py data.txt my_model

```

This command runs the script with train sequences from the file data.txt, and saves the trained model as my_model.keras



  
