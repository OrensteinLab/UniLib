import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats import pearsonr
import tensorflow as tf
import random
import os

os.chdir("../Datasets/")

seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

batch_size = 32


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


def longerOneHot(string):
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
    # read 2098 variants data with 22 barcodes and one mixed bases at least
    train = pd.read_csv("MBO_dataset.csv")
    train_sequences = list(train['101bp sequence'])  # read sequences
    train_sequences = np.array(list(map(oneHotDeg, train_sequences)))  # turn sequences to one hot vectors

    # read labels & normalize
    mean_fl_train = train['Mean_FL']
    labels_train = np.array(mean_fl_train / max(mean_fl_train))

    # use sample weights
    weights = np.array(train['total_reads'])
    weights = np.log(weights)
    weights = weights / max(weights)

    # read data for 32 validation variants
    test = pd.read_csv("yeast _32_validation_variants.csv", nrows=32, skiprows=0)
    test_sequences = list(test['sequence'])
    mean_fl_test = test['measured FL-yeast with mCore1 promoter']
    mean_fl_test = list(mean_fl_test)
    variants = list(test["variant"])

    shorter_sequences = []
    shorter_sequences_fl = []
    shorter_variant = []
    longer_sequences = []
    longer_sequences_fl = []
    longer_variants = []

    # divide validation sequences, validation variants, and mean fl values into 2 groups of different length sequences
    # There include short sURS with 3 motifs and longer sURS with 6 motifs
    for seq, fl, variant in zip(test_sequences, mean_fl_test, variants):
        # for shorter sequences with length 154
        if len(seq) == 154:
            shorter_sequences.append(seq)
            shorter_sequences_fl.append(fl)
            shorter_variant.append(variant)
        # for longer sequences with length 239
        else:
            longer_sequences.append(seq)
            longer_sequences_fl.append(fl)
            longer_variants.append(variant)

    # for longer sequences, get only 186bp sequence and exclude restrictions sites on both sides
    longer_sequences = [seq[27:213] for seq in longer_sequences]
    # for shorter sequences, get only 101bp sequence and exclude restrictions sites on both sides
    shorter_sequences = [seq[27:128] for seq in shorter_sequences]  

    # turn sequences to one hot vectors
    longer_sequences_one_hot = np.array(list(map(longerOneHot, longer_sequences)))
    shorter_sequences_one_hot = np.array(list(map(oneHotDeg, shorter_sequences)))

    # initialize a convolutional network model
    cnn_model = Sequential()
    cnn_model.add(
        Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4), use_bias=True))
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

    # predict on shorter 101bp sequences
    shorter_predictions = cnn_model.predict(shorter_sequences_one_hot)
    shorter_predictions = [pred[0] for pred in shorter_predictions]

    # save trained model's weights for all layers
    all_weights = []
    for layer in cnn_model.layers:
        layer_weights = layer.get_weights()
        all_weights.append(layer_weights)

    # initialize new models for sequences with longer input lengths
    longer_model = Sequential()
    longer_model.add(
        Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(186, 4), use_bias=True))
    longer_model.add(GlobalMaxPooling1D())
    longer_model.add(Dense(16, activation='relu'))
    longer_model.add(Dense(1, activation='linear'))
    longer_model.compile(optimizer='adam', loss='mse')

    # initialize new models weights with the weights of the trained model
    for i, layer in enumerate(longer_model.layers):
        layer.set_weights(all_weights[i])

    # use models to make predictions on longer 186bp sequence
    longer_predictions = longer_model.predict(longer_sequences_one_hot)
    longer_predictions = [pred[0] for pred in longer_predictions]

    # combine short and long sequences,labels and predictions
    all_predictions = shorter_predictions + longer_predictions
    all_labels = shorter_sequences_fl + longer_sequences_fl
    all_sequences = shorter_sequences + longer_sequences
    all_variants = shorter_variant + longer_variants

    # calculate Pearson correlation on 32 variants
    corr, p_value = pearsonr(all_predictions, all_labels)

    print("Correlations on 32 validation variants: ", corr, "P value: ", p_value)

    variant_32 = pd.DataFrame()

    # add sequences and variants columns
    variant_32['sequence'] = all_sequences
    variant_32['variant'] = all_variants

    # map variants to their #GB number
    variant_gb_mapping = dict(zip(test['variant'], test["gb #"]))

    # add gb num column according to mapping
    variant_32['gb #'] = variant_32['variant'].map(variant_gb_mapping)

    # read CHO and Hela cell data from csv file
    measured_CHO_Hela = pd.read_csv("Hela-CHO-yeast data.csv")

    # merge our table with the Hela and CHO data table
    variant_32 = pd.merge(variant_32, measured_CHO_Hela, on='gb #', how='left')

    # add mean fl and ml prediction columns to dataframe
    variant_32['mean fl'] = all_labels
    variant_32['ML prediction'] = all_predictions

    # Save the DataFrame to a new CSV file
    variant_32.to_csv("32_sURS_model_results.csv", index=False)

    # variant name of 8 uncorrelated variants between yeast and CHO cells
    variants_to_drop = [
        "(3_30_28GA)x2*",
        "(3_30_28TC)x2*",
        "3_30_28TC*",
        "3_28TC_30*",
        "28TC_3_30*",
        "28TC_30_3*",
        "30_3_28TC*",
        "30_28TC_3*"
    ]

    # Create a new DataFrame without the 8 uncorrelated dropped variants and with 24 correlated variants
    correlated_24_variants = variant_32[~variant_32['variant'].isin(variants_to_drop)].copy()

    print("Correlation between  model predictions and 24 correlated yeast-CHO variants: ",
          pearsonr(list(correlated_24_variants['measured FL-yeast with mCore1 promoter']),
                   list(correlated_24_variants["ML prediction"])))
    print("Correlation between CHO cells and model predictions: ",
          pearsonr(list(correlated_24_variants['CHO']), list(correlated_24_variants["ML prediction"])))
    print("Correlation Hela cells and model predictions: ",
          pearsonr(list(correlated_24_variants['Hela (med cell #)']), list(correlated_24_variants["ML prediction"])))

    # Save the new DataFrame to a new CSV file
    correlated_24_variants.to_csv("24_correlated_sURS_model_results.csv", index=False)


if __name__ == "__main__":
    main()
