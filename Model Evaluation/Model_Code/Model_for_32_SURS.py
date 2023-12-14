import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats import pearsonr
import tensorflow as tf
import random
import os

seed_value=42

os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

batch_size=32


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
        "N":  [0.25, 0.25, 0.25, 0.25]
    }

    # Initialize an empty matrix with the desired shape (4x101)
    one_hot_matrix = np.zeros((101, 4),dtype=np.float32)

    for i, base in enumerate(string):
        one_hot_matrix[i, :] = mapping.get(base)

    return one_hot_matrix

def longerOneHot(string):
  
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "K": [0, 0, 0.5, 0.5],
        "M": [0.5, 0.5, 0, 0],
        "N":  [0.25, 0.25, 0.25, 0.25]
    }

    # Initialize an empty matrix with the desired shape (4x101)
    one_hot_matrix = np.zeros((239, 4),dtype=np.float32)

    for i, base in enumerate(string):
        one_hot_matrix[i, :] = mapping.get(base)

    return one_hot_matrix

def shorterOneHot(string):

    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "K": [0, 0, 0.5, 0.5],
        "M": [0.5, 0.5, 0, 0],
        "N":  [0.25, 0.25, 0.25, 0.25]
    }

    # Initialize an empty matrix with the desired shape (4x101)
    one_hot_matrix = np.zeros((154, 4),dtype=np.float32)

    for i, base in enumerate(string):
        one_hot_matrix[i, :] = mapping.get(base)

    return one_hot_matrix

def main():

    # read 2435 variants data with 22 barcodes
    train = pd.read_csv("Table2435.csv")
    train_sequences = list(train['x101bpsequence'])  # read sequences
    train_sequences = np.array(list(map(oneHotDeg, train_sequences)))  # turn sequences to one hot vectors

    # read labels & normalize
    mean_fl_train = train['MeanFL']
    labels_train = np.array(mean_fl_train / max(mean_fl_train))

    # use sample weights
    weights = np.array(train['readtot'])
    weights = np.log(weights)
    weights = weights / max(weights)

    test = pd.read_csv('32_sURS.csv')
    test_sequences=list(test['sequence'])
    mean_fl_test=test['mean_fl']
    mean_fl_test=list(mean_fl_test)

    shorter_sequences=[]
    shorter_sequences_fl=[]
    longer_sequences=[]
    longer_sequences_fl = []

    # divide validation sequences into 2 groups of different lengths
    for seq,fl in zip(test_sequences,mean_fl_test):
        if len(seq)==154:
            shorter_sequences.append(seq)
            shorter_sequences_fl.append(fl)
        else:
            longer_sequences.append(seq)
            longer_sequences_fl.append(fl)

    longer_sequences_one_hot=np.array(list(map(longerOneHot, longer_sequences)))
    shorter_sequences_one_hot = np.array(list(map(shorterOneHot, shorter_sequences)))

    # initialize a convolutional network model
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4), use_bias=True))
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(16, activation='relu'))
    cnn_model.add(Dense(1, activation='linear'))
    cnn_model.compile(optimizer='adam', loss='mse')

    shuffled_indices = np.random.permutation(range(len(train_sequences)))

    # Use the shuffled indices to access data while keeping their relative order
    sequences_shuffled = train_sequences[shuffled_indices]
    labels_shuffled = labels_train[shuffled_indices]
    weights_shuffled = weights[shuffled_indices]

    # Fit the model on shuffled data
    cnn_model.fit(sequences_shuffled, labels_shuffled, epochs=3, batch_size=batch_size, verbose=1,
                         sample_weight=weights_shuffled, shuffle=True)

    # save trained model's weights for all layers
    all_weights = []
    for layer in cnn_model.layers:
        layer_weights = layer.get_weights()
        all_weights.append(layer_weights)

    # initialize new models for sequences of different lengths
    longer_model = Sequential()
    longer_model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(239, 4), use_bias=True))
    longer_model.add(GlobalMaxPooling1D())
    longer_model.add(Dense(16, activation='relu'))
    longer_model.add(Dense(1, activation='linear'))
    longer_model.compile(optimizer='adam', loss='mse')

    shorter_model = Sequential()
    shorter_model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(154, 4), use_bias=True))
    shorter_model.add(GlobalMaxPooling1D())
    shorter_model.add(Dense(16, activation='relu'))
    shorter_model.add(Dense(1, activation='linear'))
    shorter_model.compile(optimizer='adam', loss='mse')

    # initialize new models weights with the weights of the trained model
    for i, layer in enumerate(longer_model.layers):
        layer.set_weights(all_weights[i])

    for i, layer in enumerate(shorter_model.layers):
        layer.set_weights(all_weights[i])

    # use models to make predictions on different lengths sequences
    longer_predictions=longer_model.predict(longer_sequences_one_hot)
    longer_predictions=[pred[0] for pred in longer_predictions]

    shorter_predictions=shorter_model.predict(shorter_sequences_one_hot)
    shorter_predictions=[pred[0] for pred in shorter_predictions]

    # calculate Pearson correlation on 32 variants
    corr, p_value = pearsonr(shorter_predictions+longer_predictions, shorter_sequences_fl+longer_sequences_fl)

    print("Correlations on 32: ", corr, "P value: ", p_value)

    variant_32=pd.DataFrame()

    variant_32['sequence']=shorter_sequences+longer_sequences

    pbm = pd.read_csv("measured yeast all validation results.csv")

    sequence_variant_mapping = dict(zip(pbm['sequence'], pbm['variant']))

    variant_32['variant'] = variant_32['sequence'].map(sequence_variant_mapping)

    variant_32['mean fl']=shorter_sequences_fl+longer_sequences_fl
    variant_32['ML prediction']=shorter_predictions+longer_predictions

    # Save the DataFrame to a new CSV file
    variant_32.to_csv("32_sURS_model_results.csv", index=False)

if __name__ == "__main__":
    main()
