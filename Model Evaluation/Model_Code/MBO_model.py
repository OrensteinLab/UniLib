import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats import pearsonr
import tensorflow as tf
import random
import os

seed_value=21

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


def train_predict_300_test():

    # read 300 validation variants from csv file
    test = pd.read_csv("300_test_variants.csv")
    test_sequences = list(test['101bp sequence'])  # read sequences
    test_sequences = np.array(list(map(oneHotDeg, test_sequences)))  # use one hot function
    # read labels
    test_labels =test['Mean_FL']

    # read 2098 variants data with mixed bases and 22 barcodes
    train = pd.read_csv("MBO_dataset.csv")
    ##  remove 300 test sequences from all sequences
    train = train[~train['101bp sequence'].isin(list(test['101bp sequence']))]
    train_sequences = list(train['101bp sequence'])  # read sequences
    train_sequences = np.array(list(map(oneHotDeg, train_sequences)))  # turn sequences to one hot vectors

    # read labels & normalize
    mean_fl_train = train['Mean_FL']
    labels_train = np.array(mean_fl_train / max(mean_fl_train))

    # define sample weights
    weights = np.array(train['total_reads'])
    weights = np.log(weights)
    weights = weights / max(weights)

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
    labels_shuffled = labels_train[shuffled_indices]
    weights_shuffled = weights[shuffled_indices]

    # Fit the model on shuffled data
    cnn_model.fit(sequences_shuffled, labels_shuffled, epochs=3, batch_size=batch_size, verbose=1,
                  sample_weight=weights_shuffled, shuffle=True)

    predictions = cnn_model.predict(test_sequences)
    predictions = [pred[0] for pred in predictions]

    correlation,p_value= pearsonr(predictions,test_labels)

    print("Correlation on 300 test: ",correlation, " p value: ",p_value)

    test["MBO_model_prediction"] = predictions

    test.to_csv('MBO_predictions_test_300.csv', index=False)



def ensemble_model(train_sequences, train_labels, train_weights, test_sequences):
    """
    Run an ensemble of machine learning models and make predictions on test sequences.

    Parameters:
    - train_sequences: Training DNA sequences.
    - train_labels: Labels for the training data.
    - train_weights: Weights for the training data.
    - test_sequences: Test DNA sequences (300 variants).

    Returns:
    - predictions (list): Predictions for the 11 variants.
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

    predictions = cnn_model.predict(test_sequences)
    predictions=[pred[0] for pred in predictions]

    return predictions


def main():

    ## train model generate predictions for 300 test variants
    train_predict_300_test()

    # read 11 validation variants
    validation = pd.read_csv('11_validation_variants.csv')
    validation_sequences = list(validation['sequence'])  # read sequences
    # exclude 15-nt barcode from the variant sequence
    validation_sequences= [sequence[15:] for sequence in validation_sequences]
    validation_sequences = np.array(list(map(oneHotDeg, validation_sequences)))  # turn to one hot vectors
    # read labels
    validation_labels = validation['mean_fl']

    # read 2098 variants data with mixed bases and 22 barcodes
    train = pd.read_csv("MBO_dataset.csv")
    train_sequences = list(train['101bp sequence'])  # read sequences
    train_sequences = np.array(list(map(oneHotDeg, train_sequences)))  # turn sequences to one hot vectors

    # read labels & normalize
    mean_fl_train = train['Mean_FL']
    labels_train = np.array(mean_fl_train/ max(mean_fl_train))

    # define sample weights
    weights = np.array(train['total_reads'])
    weights = np.log(weights)
    weights = weights / max(weights)

    all_predictions_val = []

    # run 100 models with random initialization as part of the random ensemble initialization technique
    for i in range(100):
        # Use the function to train model on MBO dataset with random initializations and make predictions
        predictions = ensemble_model(train_sequences, labels_train, weights,validation_sequences)
        all_predictions_val.append(predictions)

    # calculate mean over the predictions of the 100 ensemble models
    avg_predictions_val = np.mean(all_predictions_val, axis=0)

    # calculate pearson correlation on validation sets
    corr,p_value  = pearsonr(avg_predictions_val, validation_labels)

    # print pearson correlation & p-value
    print("Correlations on 11: ", corr, "P value: ",p_value)

    # create columns for the 100 models average predictions and the true labels
    validation["Average_MBO_prediction"] = avg_predictions_val

    validation.to_csv('MBO_predictions_11_validation.csv', index=False)

if __name__ == "__main__":
    main()
