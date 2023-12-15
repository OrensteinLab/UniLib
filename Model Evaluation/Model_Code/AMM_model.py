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


def train_predict(train_sequences, train_labels, train_weights, test_data1, test_data2):
    """
    Run an ensemble of machine learning models and make predictions on test sequences.

    Parameters:
    - train_sequences: Training DNA sequences.
    - train_labels: Labels for the training data.
    - train_weights: Weights for the training data.
    - test_sequences1: Test DNA sequences (300 variants).
    - comp_test_sequences1: Reverse complement of test sequences 1.
    - test_sequences2: Test DNA sequences (11 variants).
    - comp_test_sequences2: Reverse complement of test sequences 2.

    Returns:
    - predictions300 (list): Predictions for the 300 variants.
    - predictions11 (list): Predictions for the 11 variants.
    """

    # Initialize a new convolutional network model
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4), use_bias=True))
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(16, activation='relu'))
    cnn_model.add(Dense(1, activation='linear'))
    cnn_model.compile(optimizer='adam', loss='mse')


    # Shuffle the data
    shuffled_indices = np.random.permutation(range(len(train_sequences)))

    # Use the shuffled indices to access data while keeping their relative order
    sequences_shuffled = train_sequences[shuffled_indices]
    labels_shuffled = train_labels[shuffled_indices]
    weights_shuffled = train_weights[shuffled_indices]

    # Fit the model on shuffled data
    cnn_model.fit(sequences_shuffled, labels_shuffled, epochs=3, batch_size=batch_size, verbose=1,
                         sample_weight=weights_shuffled, shuffle=True)

    predictions1 = cnn_model.predict(test_data1)
    predictions1=[pred[0] for pred in predictions1]
    predictions2 = cnn_model.predict(test_data2)
    predictions2 = [pred[0] for pred in predictions2]

    return predictions1, predictions2



def main():

    # read 300 validation variants
    test1 = pd.read_csv("300_test_variants.csv")
    test_sequences1 = list(test1['101bp sequence'])  # read sequences
    test_sequences1 = np.array(list(map(oneHotDeg, test_sequences1)))  # use one hot function
    # read labels
    test_labels1 =test1['Mean_FL']


    # read 11 validation variants
    test2 = pd.read_csv('11_validation_variants.csv')
    test_sequences2 = list(test2['sequence'])  # read sequences
    # exclude 15-nt barcode from the variant sequence
    test_sequences2 = [sequence[15:] for sequence in test_sequences2]
    test_sequences2 = np.array(list(map(oneHotDeg, test_sequences2)))  # turn to one hot vectors
    # read labels
    test_labels2 = test2['mean_fl']

    # read 2432 variants data with 22 barcodes
    train = pd.read_csv("train_set_variants_22_barcodes.csv")
    train_sequences = list(train['101bp sequence'])  # read sequences
    train_sequences = np.array(list(map(oneHotDeg, train_sequences)))  # turn sequences to one hot vectors

    # read labels & normalize
    mean_fl_train = train['Mean_FL']
    labels_train = np.array(mean_fl_train/ max(mean_fl_train))

    # define sample weights
    weights = np.array(train['total_reads'])
    weights = np.log(weights)
    weights = weights / max(weights)


    all_predictions1 = []
    all_predictions2 = []

    # run 100 models with random initialization as part of the random ensemble initialization technique
    for i in range(100):
        # Use the function to train model on MBO dataset with random initializations and make predictions
        predictions1, predictions2 = train_predict(train_sequences, labels_train, weights,test_sequences1,test_sequences2)
        all_predictions1.append(predictions1)
        all_predictions2.append(predictions2)

    # calculate mean over the predictions of the 100 ensemble models
    avg_predictions1 = np.mean(all_predictions1, axis=0)
    avg_predictions2 = np.mean(all_predictions2, axis=0)

    # calculate pearson correlation on validation sets
    corr1,p_value1  = pearsonr(avg_predictions1, test_labels1)
    corr2,p_value2 = pearsonr(avg_predictions2, test_labels2)

    # print pearson correlation & p-value
    print("Correlations on 11: ", corr2, "P value: ",p_value2)
    print("Correlations on 300: ", corr1, "P value: ",p_value1)

    # create columns for the 100 models average predictions and the true labels
    test1["Average_model_prediction"] = avg_predictions1
    test2["Average_model_prediction"] = avg_predictions2
    test1["True_labels"] = test_labels1
    test2["True_labels"] = test_labels2

    # Save the DataFrame with true labels verses predictions to a CSV file
    test1.to_csv('test_300_predictions_AMM.csv', index=False)
    test2.to_csv('test_11_predictions_AMM.csv', index=False)

if __name__ == "__main__":
    main()
