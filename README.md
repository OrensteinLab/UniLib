## UNILIB ##

# Introduction

In the UNILIB study, we used data from a Massive Parallel Reported Assay (MPRA) in that generated expression data for Sythetic Upstream Regulatory Regions (sURS) containing various TFFBS in yeast.The goal of the study was to develop an algorithm of sURS design for boosting gene expression in yeast and to build machine learning models that would be able to learn the variant data from the experiment and make predictions on new sURS sequences
This repository includes all of the scripts and trained ML models used in the UNILIB study. In addition, the repository includes a scripts for making predictions on new DNA sequence files using a chosen ML model and to train an ML on a DNA sequence file


# Setup environment

```
# create virtual conda environment
conda create -n Unilib python=3.9
conda activate Unilib

git clone https://github.com/OrensteinLab/UniLib.git

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

**Prerequisites**


Before using the script, ensure that the following dependencies are installed:

* argparse: A module for parsing command-line arguments.<br>
* pandas: A powerful data manipulation library.<br>
* keras: A high-level neural networks API.<br>
* numpy: A fundamental package for scientific computing with Python.<br>
* scipy: A library for scientific computing and statistical routines.<br>
<br>

**Usage**
<br>

Execute the script from the command line using the following syntax:<br>

```

python predict.py model_name output_file input_file

```

* model_name: Specify the model to be used for predictions (ADM, AMM, or default MBO).<br>
* output_file: The name of the file where predictions will be saved.<br>
* input_file: The name of the file containing input DNA sequences.<br>


**DNA Sequence Encoding**
  

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


**Input File Format**


The input file should contain DNA sequences of synthetic upstream regulatory region, with each sequence on a new line. The sequence should be 101 bases in length for the models. The script reads these sequences from the input file.


**Output**

The script outputs predictions for each input sequence to the specified output_file. Each prediction is written to a new line.

**Example**

<br>


```

python predict.py ADM predictions.txt input_sequences.txt

```

This command runs the script using the ADM model, with input sequences from the file input_sequences.txt, and saves the predictions to the file predictions.txt.



  
