import argparse
import pandas as pd
from keras.models import load_model
import numpy as np
from scipy.stats import pearsonr

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

    # load the chosen model
    if model_name=="ADM":
        model=load_model("ADM.h5")
    elif model_name=="AMM":
        model=load_model("AMM.h5")
    else: # default MBO model
        model=load_model("MBO.h5")

    sequences=[]
    # read sequences from input file
    with open(input_file,"r") as file:
        for line in file.readlines():
            sequences.append(line)
            
    # ensure sequences are no longer than 101 nt
    sequences=[seq if len(seq)<=101 else seq[:101] for seq in sequences]

    # turn sequences to One Hot vectors
    sequences=np.array(list(map(oneHotDeg,sequences)))

    # use model to make predictions on sequences
    predictions=[pred[0] for pred in model.predict(sequences)]

    with open(output_file,"w") as output:
        for pred in predictions:
            output.write(str(pred)+"\n")

if __name__ == "__main__":
    main()












