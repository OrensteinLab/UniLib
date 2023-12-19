# DNA Sequence Prediction Readme

## Overview ##


This Python script enables the prediction of numerical expression values for given DNA sequences using pre-trained deep learning models. The models: ADM, AMM, and MBO, predict mean flourescence based on the dna sequence of synthetic upstream regulatory region. This readme provides information on how to use the script, prerequisites for execution, and details about the models and input data.

**Prerequisites**


Before using the script, ensure that the following dependencies are installed:

* argparse: A module for parsing command-line arguments.<br>
* pandas: A powerful data manipulation library.<br>
* keras: A high-level neural networks API.<br>
* numpy: A fundamental package for scientific computing with Python.<br>
* scipy: A library for scientific computing and statistical routines.<br>

**Usage**


Execute the script from the command line using the following syntax:<br>

python predict.py model_name output_file input_file<br>

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

* ADM (All Data Model) - Trained on 20,000 sequences with the heighest number of read from the expreiment<br>
* AMM (All Motif Model)- Trained on 2,435 sequences with 22 barcodes each <br>
* MBO (Mixed Bases Only Model)- Trained on 2,098 sequences with 22 barcodes and at least one mixed base (K/M) <br>

**Input File Format**


The input file should contain DNA sequences of synthetic upstream regulatory region, with each sequence on a new line. The sequence should be 101 bases in length for the models. The script reads these sequences from the input file.

**Output**


The script outputs predictions for each input sequence to the specified output_file. Each prediction is written to a new line.

**Example**


python predict.py ADM predictions.txt input_sequences.txt<br>

This command runs the script using the ADM model, with input sequences from the file input_sequences.txt, and saves the predictions to the file predictions.txt.

