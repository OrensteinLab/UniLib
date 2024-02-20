import argparse
from keras.models import load_model
import numpy as np
import os
import pandas as pd
import re

def custom_padding(sequence, desired_length=186):
    # custom padding funcntions
    sequence = list(sequence)
    padding_length = max(0, desired_length - len(sequence))

    # Calculate padding on both sides
    left_padding = padding_length // 2
    right_padding = padding_length - left_padding

    # Add padding to both sides of the sequence
    padded_sequence = ["N"]*left_padding + sequence + ["N"]*right_padding

    padded_sequence= ''.join(padded_sequence)

    return padded_sequence

# Handling ambiguous bases in motifs
def handle_ambiguous_bases(motif):
    motif = motif.replace('K', '[GT]')
    motif = motif.replace('M', '[AC]')
    return motif


def motifs_present(sequences, motifs,regex=False):
    # handle ambiguous bases with regular expressions because test does not contain K and M
    if regex:
        motifs = [handle_ambiguous_bases(motif) for motif in motifs]
    motifs_vectors = []
    for seq in sequences:
        motifs_count = [0 for _ in range(41)]
        for i, motif in enumerate(motifs):
            if regex:
                matches = re.finditer(motif, seq)
                count = sum(1 for _ in matches)
            else:
                count = seq.count(motif)
            motifs_count[i] += count
        motifs_count = np.array(motifs_count)
        motifs_vectors.append(motifs_count)

    motifs_vectors = np.array(motifs_vectors)

    return motifs_vectors

def oneHotDeg(string):
    """
    Converts a DNA sequence to a one-hot encoding.
    Parameters:
    - string (str): Input DNA sequence.
    Returns:
    - one_hot_matrix (numpy.ndarray): One-hot encoding of the input DNA sequence.
    """
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "K": [0, 0, 0.5, 0.5],
        "M": [0.5, 0.5, 0, 0],
        "N": [0.25, 0.25, 0.25, 0.25]
    }

    # Initialize an empty matrix with the desired shape (4x101)
    one_hot_matrix = np.zeros((101, 4), dtype=np.float32)

    for i, base in enumerate(string):
        one_hot_matrix[i, :] = mapping.get(base)

    return one_hot_matrix

def longerOneHotDeg(string):
    """
    Converts a DNA sequence to a one-hot encoding.
    Parameters:
    - string (str): Input DNA sequence.
    Returns:
    - one_hot_matrix (numpy.ndarray): One-hot encoding of the input DNA sequence.
    """
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "K": [0, 0, 0.5, 0.5],
        "M": [0.5, 0.5, 0, 0],
        "N": [0.25, 0.25, 0.25, 0.25]
    }

    # Initialize an empty matrix with the desired shape (4x101)
    one_hot_matrix = np.zeros((186, 4), dtype=np.float32)

    for i, base in enumerate(string):
        one_hot_matrix[i, :] = mapping.get(base)

    return one_hot_matrix


def main():
    # parse cmd arguments
    parser = argparse.ArgumentParser(description='Run model')
    parser.add_argument('model_name', type=str, help='model name')
    parser.add_argument('output_file', type=str, help='output file name')
    parser.add_argument('input_file', type=str, help='Input file name')

    args = parser.parse_args()

    # Extract input arguments
    model_name = args.model_name
    output_file = args.output_file
    input_file = args.input_file

    sequences = []
    # read sequences from input file
    with open(input_file, "r") as file:
        for line in file.readlines():
            sequences.append(line)

    # ensure sequences are no longer than 101 nt
    sequences = [seq if len(seq) <= 101 else seq[:101] for seq in sequences]

    # load the chosen model
    if model_name=="ADM":
        model=load_model("Model_Evaluation/saved_models/ADM.h5")
    elif model_name=="AMM":
        model=load_model("Model_Evaluation/saved_models/AMM.h5")
    elif model_name=="MLAM":
        model = load_model("Model_Evaluation/saved_models/MLAM.h5")
    else: # default MBO model
        model=load_model("Model_Evaluation/saved_models/MBO.h5")

    if model_name=="MLAM": # for ML additive model

        # read all motifs from table
        motifs_table = pd.read_csv("Model Evaluation/Datasets/Unilib_Motifs_info.csv")
        motifs = list(motifs_table['Motif sequence'])  # read motifs from file
        motifs.remove('GAATATTCTAGAATATTC')  # remove rare motif

        motifs_present_pred=motifs_present(sequences,motifs,regex=True)

        padded_sequences=[custom_padding(sequence) for sequence in sequences]

        # turn sequences to One Hot vectors
        one_hot_sequences = np.array(list(map(longerOneHotDeg, padded_sequences)))

        predictions = [pred[0] for pred in model.predict([one_hot_sequences,motifs_present_pred])]

    else: # not using MLAM model
        # turn sequences to One Hot vectors
        one_hot_sequences = np.array(list(map(oneHotDeg, sequences)))
        # use model to make predictions on sequences
        predictions=[pred[0] for pred in model.predict(one_hot_sequences)]

    # write prediction results to output file
    with open(output_file,"w") as output:
        for pred in predictions:
            output.write(str(pred)+"\n")

if __name__ == "__main__":
    main()












