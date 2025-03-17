import argparse
import tensorflow as tf
import numpy as np
from keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

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
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_file', type=str, help='data')
    parser.add_argument('model_name', type=str, help='model')

    args = parser.parse_args()

    # Extract input arguments
    data_file = args.data_file
    model_name = args.model_name

    sequences = []
    labels = []
    # read sequences from input file
    with open(data_file, "r") as file:
        for line in file.readlines():
            seq, label = line.strip().split()
            label = float(label)
            sequences.append(seq)
            labels.append(label)

    # ensure sequences are no longer than 101 nt
    sequences = [seq if len(seq) <= 101 else seq[:101] for seq in sequences]

    # normalize labels
    max_label = max(labels)
    labels = [label/max_label for label in labels]
    labels=np.array(labels)

    # turn sequences to One Hot vectors
    sequences = np.array(list(map(oneHotDeg, sequences)))

    # create convolution network model
    cnn_model = Sequential()
    cnn_model.add(
        Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4), use_bias=True))
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(16, activation='relu'))
    cnn_model.add(Dense(1, activation='linear'))
    cnn_model.compile(optimizer='adam', loss='mse')

    # fit model on training data
    cnn_model.fit(sequences, labels, epochs=5, batch_size=32, verbose=1, shuffle=True)
    # save trained model
    cnn_model.save(str(model_name)+".keras")

    print("Model saved successfully as: " + str(model_name) + ".keras")


if __name__ == "__main__":
    main()
