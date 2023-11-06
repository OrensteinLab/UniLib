import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats import pearsonr
from tensorflow.keras.models import load_model, save_model
import tensorflow as tf
from model import Model

tf.random.set_seed(42)
np.random.seed(42)


class Model:

    def __init__(self):
        """
        Initialize the neural network model architecture.
        """
        self.cnn_model = Sequential()
        self.cnn_model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4), use_bias=True))
        self.cnn_model.add(GlobalMaxPooling1D())
        self.cnn_model.add(Dense(16, activation='relu'))
        self.cnn_model.add(Dense(1, activation='linear'))
        self.cnn_model.compile(optimizer='adam', loss='mse')

    def fit(self, sequences, labels, weights, epochs):

        # Shuffle sequences and labels
        shuffled_indices = np.arange(len(sequences))
        np.random.shuffle(shuffled_indices)
        sequences = sequences[shuffled_indices]
        labels = labels[shuffled_indices]

        if weights is not None:
            weights = weights[shuffled_indices]
            # fit model on data
            self.cnn_model.fit(sequences, labels, epochs=epochs, batch_size=32, verbose=1, sample_weight=weights)
        else:
            # fit model on data
            self.cnn_model.fit(sequences, labels, epochs=epochs, batch_size=32, verbose=1)

    def predict(self, test):

        return self.cnn_model.predict(test)

    def save(self, file_name):

        self.cnn_model.save(file_name)

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
        "M": [0.5, 0.5, 0, 0]
    }

    # Initialize an empty matrix with the desired shape (4x101)
    one_hot_matrix = np.zeros((101, 4),dtype=np.float32)

    for i, base in enumerate(string):
        one_hot_matrix[i, :] = mapping.get(base, [0.25, 0.25, 0.25, 0.25])

    return one_hot_matrix


def reverse_complement(dna_sequence):
    """
    Computes the reverse complement of a given DNA sequence.

    Parameters:
    - dna_sequence (str): Input DNA sequence.

    Returns:
    - reverse_comp_sequence (str): Reverse complement of the input DNA sequence.
    """
    # Define a dictionary to map nucleotides to their complements
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N', 'M': 'K', 'K': 'M'}

    # Reverse the DNA sequence and find the complement for each nucleotide
    reverse_comp_sequence = [complement_dict[nt] for nt in reversed(dna_sequence)]

    # Convert the list of complement nucleotides back to a string
    return ''.join(reverse_comp_sequence)


def reverse_comp_prediction(model, test_data):
    """
    Make predictions on both the original sequences and their reverse complements and average the results.

    Parameters:
    - model: The machine learning model.
    - test_sequences: Original DNA sequences.
    - comp_test_sequences: Reverse complement DNA sequences.

    Returns:
    - predictions (list): Average predictions for each sequence.
    """
    test_sequences, comp_test_sequences=test_data

    # use model to predict test sequences labels test data
    predictions_original = model.predict(test_sequences)
    # use model to predict reverse complement sequences labels
    predictions_reverse_comp = model.predict(comp_test_sequences)
    predictions = []
    # for each sequence, the model's prediction is the average of the model's prediction on the sequence itself and its reverse complement
    for pred, comp_pred in zip(predictions_original, predictions_reverse_comp):
        avg_pred = (pred[0] + comp_pred[0]) / 2
        predictions.append(avg_pred)

    return predictions


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

    # Load the pretrained model's weights
    pretrained_model = load_model('/content/pretrained_cnn_model.h5')

    # Shuffle the data
    shuffled_indices = np.arange(len(train_sequences))
    np.random.shuffle(shuffled_indices)

    # Use the shuffled indices to access data while keeping their relative order
    sequences_shuffled = train_sequences[shuffled_indices]
    labels_shuffled = train_labels[shuffled_indices]
    weights_shuffled = train_weights[shuffled_indices]

    # Fit the model on shuffled data
    pretrained_model.fit(sequences_shuffled, labels_shuffled, epochs=3, batch_size=32, verbose=1,
                         sample_weight=weights_shuffled, shuffle=True)

    predictions1 = reverse_comp_prediction(pretrained_model, test_data1)
    predictions2 = reverse_comp_prediction(pretrained_model, test_data2)

    return predictions1, predictions2


def main():

    # read csv of 6 million reads from de Boer-Regev experiment
    train1 = pd.read_csv("/content/6_million_read.csv")

    sequences1 = list(train1['Sequence'])  # read sequences
    reverse_complement1 = list(map(reverse_complement, sequences1))  # create reverse complement sequences
    sequences1.extend(reverse_complement1)  # add reverse complements to sequences
    sequences1 = np.array(list(map(oneHotDeg, sequences1)))  # use the one-hot function on the sequences

    # read labels & normalize
    mean_fl1 = train1['Mean_Fl']
    labels1 = np.array(mean_fl1 / max(mean_fl1))  # divide each expression by the max
    labels1 = np.concatenate((labels1, labels1))


    # read 67k variants data from csv file
    train2 = pd.read_csv("/content/all_variants_without_test.csv")

    sequences2 = list(train2['VariableRegion'])  # read sequences
    reverse_complement2 = list(map(reverse_complement, sequences2))  # create reverse complement sequences
    sequences2.extend(reverse_complement2)  # add reverse complement sequences to sequences
    sequences2 = np.array(list(map(oneHotDeg, sequences2)))  # turn sequences to one hot vectors using function

    # read labels & normalize
    mean_fl2 = train2['Mean_FL']
    labels2 = np.array(mean_fl2 / max(mean_fl2))  # divide expression by max expression
    labels2 = np.concatenate((labels2, labels2))

    # use sample weights
    weights2 = np.array(train2['total_reads'])
    weights2 = np.log(weights2)
    weights2 = weights2 / max(weights2)
    weights2 = np.concatenate((weights2, weights2))


    # read 2135 variants data with 22 barcodes
    train3 = pd.read_csv("/content/train_set_variants_20_barcodes.csv")

    sequences3 = list(train3['VariableRegion'])  # read sequences
    reverse_complement3 = list(map(reverse_complement, sequences3))  # reverse complement
    sequences3.extend(reverse_complement3)  # add reverse complements to sequences
    sequences3 = np.array(list(map(oneHotDeg, sequences3)))  # turn sequences to one hot vectors

    # read labels & normalize
    mean_fl3 = train3['Mean_FL']
    labels3 = np.array(mean_fl3 / max(mean_fl3))
    labels3 = np.concatenate((labels3, labels3))

    # use sample weights
    weights3 = np.array(train3['total_reads'])
    weights3 = np.log(weights3)
    weights3 = weights3 / max(weights3)
    weights3 = np.concatenate((weights3, weights3))


    # read 300 validation variants
    test1 = pd.read_csv("/content/300_test_variants.csv")

    test_sequences1 = list(test1['VariableRegion'])  # read sequences
    comp_test_sequences1 = list(map(reverse_complement, test_sequences1))  # reverse complements
    test_sequences1 = np.array(list(map(oneHotDeg, test_sequences1)))  # use one hot function
    comp_test_sequences1 = np.array(list(map(oneHotDeg, comp_test_sequences1)))
    test_data1=test_sequences1,comp_test_sequences1

    # read & normalize labels
    mean_fl_test1 = test1['Mean_FL']
    test_labels1 = np.array(mean_fl_test1 / max(mean_fl_test1))


    # read 11 validation variants
    test2 = pd.read_csv('/content/11_validation_variants.csv')
    test_sequences2 = list(test2['barcoded variant'])  # read sequences

    # exclude 15-nt barcode from the variant sequence
    sliced_sequences = []
    # Iterate through each sequence and ignore first 15 nucleotides
    for sequence in test_sequences2:
        sliced_sequence = sequence[15:]  # Slice the sequence from index 15 onwards
        sliced_sequences.append(sliced_sequence)

    test_sequences2 = sliced_sequences
    comp_test_sequences2 = list(map(reverse_complement, test_sequences2))  # use reverse complement function
    test_sequences2 = np.array(list(map(oneHotDeg, test_sequences2)))  # turn to one hot vectors
    comp_test_sequences2 = np.array(list(map(oneHotDeg, comp_test_sequences2)))  # turn to one hot vectors
    test_data2 = test_sequences2,comp_test_sequences2

    # read & normalize labels
    mean_fl_test2 = test2['yeast average']
    test_labels2 = np.array(mean_fl_test2 / max(mean_fl_test2))


    # create a convolutional network model
    cnn_model= Model()
    #fit on 6 million sequences
    cnn_model.fit(sequences=sequences1,labels=labels1,weights=None,epochs=1)
    # fit of 67k sequences
    cnn_model.fit(sequences=sequences2, labels=labels2, weights=weights2, epochs=3)
    # save pretrained model's weights
    cnn_model.save('pretrained_cnn_model.h5')


    all_predictions1 = []
    all_predictions2 = []

    # run 100 models as part of the random ensemble initialization technique
    for i in range(100):
        # Use function to train the pretrained model on 2,135 variants with 22 barcodes (with random intialization) and to make predictions
        predictions1, predictions2 = train_predict(sequences3, labels3, weights3,test_data1,test_data2)
        all_predictions1.append(predictions1)
        all_predictions2.append(predictions2)

    # calculate mean over the predictions of the 100 models
    avg_predictions1 = np.mean(all_predictions1, axis=0)
    avg_predictions2 = np.mean(all_predictions2, axis=0)

    # calculate pearson correlation on validation sets
    corr1,p_value1  = pearsonr(avg_predictions1, test_labels1)
    corr2,p_value2 = pearsonr(avg_predictions2, test_labels2)

    # print pearson correlation
    print("Correlations on 11: ", corr2, "P value: ",p_value2)
    print("Correlations on 300: ", corr1, "Pvalue: ",p_value1)

    # add columns for the 100 models average predictions and the true labels to the csv files of the validation sets
    test1["Average_model_prediction"] = avg_predictions1
    test2["Average_model_prediction"] = avg_predictions2
    test1["True_labels"] = test_labels1
    test2["True_labels"] = test_labels2

    # Save the DataFrame to a CSV file
    test1.to_csv("300_validation_with_predictions.csv', index=False)
    test2.to_csv('11_validation_with_predictions.csv', index=False)

if __name__ == "__main__":
    main()
