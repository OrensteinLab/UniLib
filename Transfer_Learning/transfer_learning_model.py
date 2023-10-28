import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats import pearsonr
from tensorflow.keras.models import load_model, save_model

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
    one_hot_matrix = np.zeros((101, 4), dtype=np.float32)

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


def reverse_comp_prediction(model, test_sequences, comp_test_sequences):
    """
    Make predictions on both the original sequences and their reverse complements and average the results.

    Parameters:
    - model: The machine learning model.
    - test_sequences: Original DNA sequences.
    - comp_test_sequences: Reverse complement DNA sequences.

    Returns:
    - predictions (list): Average predictions for each sequence.
    """
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


def run_ensemble(train_sequences, train_labels, train_weights, test_sequences1, comp_test_sequences1, test_sequences2,
                 comp_test_sequences2):
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

    # Load the pretrained model
    pretrained_model = load_model('pretrained_cnn_model.h5')

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

    predictions300 = reverse_comp_prediction(pretrained_model, test_sequences1, comp_test_sequences1)
    predictions11 = reverse_comp_prediction(pretrained_model, test_sequences2, comp_test_sequences2)

    return predictions300, predictions11

# read csv with 6 million from de Boer-Regev experiment
train1 = pd.read_csv("6_million_reads.csv")
sequences1 = list(train1['Sequence'])
# create reverse complement sequences
reverse_complement1 = list(map(reverse_complement, sequences1))
sequences1.extend(reverse_complement1)
sequences1 = np.array(list(map(oneHotDeg, sequences1)))
mean_fl1 = train1['Mean_Fl']
labels1 = np.array(mean_fl1 / max(mean_fl1))
labels1 = np.concatenate((labels1, labels1))

# read 67k variant data from csv file
train2 = pd.read_csv("all_variants_without_test.csv")
weights2 = np.array(train2['total_reads'])
weights2 = np.log(weights2)
weights2 = weights2 / max(weights2)
weights2 = np.concatenate((weights2, weights2))
sequences2 = list(train2['VariableRegion'])
reverse_complement2 = list(map(reverse_complement, sequences2))  # reverse complement sequences
sequences2.extend(reverse_complement2)
sequences2 = np.array(list(map(oneHotDeg, sequences2)))  # turn sequences to one hot vectors
mean_fl2 = train2['Mean_FL']
labels2 = np.array(mean_fl2 / max(mean_fl2))
labels2 = np.concatenate((labels2, labels2))

# read 2135 variants data with 20 barcodes
train3 = pd.read_csv("train_set_variants_22_barcodes.csv")
weights3 = np.array(train3['total_reads'])
weights3 = np.log(weights3)
weights3 = weights3 / max(weights3)
weights3 = np.concatenate((weights3, weights3))
sequences3 = list(train3['VariableRegion'])
reverse_complement3 = list(map(reverse_complement, sequences3))  # reverse complement
sequences3.extend(reverse_complement3)  # add reverse complements to sequences
sequences3 = np.array(list(map(oneHotDeg, sequences3)))  # turn sequences to one hot vectors
mean_fl3 = train3['Mean_FL']
labels3 = np.array(mean_fl3 / max(mean_fl3))
labels3 = np.concatenate((labels3, labels3))

# read 300 test variants
test1 = pd.read_csv("300_test_variants.csv")
test_sequences1 = list(test1['VariableRegion'])
comp_test_sequences1 = list(map(reverse_complement, test_sequences1))  # reverse complements
test_sequences1 = np.array(list(map(oneHotDeg, test_sequences1)))
comp_test_sequences1 = np.array(list(map(oneHotDeg, comp_test_sequences1))) # turn to one hot vectors
mean_fl_test1 = test1['Mean_FL']
test_labels1 = np.array(mean_fl_test1 / max(mean_fl_test1))

# read 11 validation variants
test2 = pd.read_csv('11_validation_variants.csv')
test_sequences2 = list(test2['barcoded variant'])
# exclude 15-nt barcode from the variant sequence
sliced_sequences = []
# Iterate through each sequence and ignore first 15 nucleotides
for sequence in test_sequences2:
    sliced_sequence = sequence[15:]  # Slice the sequence from index 15 onwards
    sliced_sequences.append(sliced_sequence)
test_sequences2 = sliced_sequences
comp_test_sequences2 = list(map(reverse_complement, test_sequences2))  # use reverse complement function
test_sequences2 = np.array(list(map(oneHotDeg, test_sequences2)))
comp_test_sequences2 = np.array(list(map(oneHotDeg, comp_test_sequences2)))
mean_fl_test2 = test2['yeast average']
test_labels2 = np.array(mean_fl_test2 / max(mean_fl_test2))

# create a convolution network machine learning model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4), use_bias=True))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(16, activation='relu'))
cnn_model.add(Dense(1, activation='linear'))
cnn_model.compile(optimizer='adam', loss='mse')

# fit model on 6 million variant data
cnn_model.fit(sequences1, labels1, epochs=1, batch_size=32, verbose=1)

# fit model on 67k variant data
cnn_model.fit(sequences2, labels2, epochs=3, shuffle=True, batch_size=32, verbose=1, sample_weight=weights2)

# save the pretrained model's weights
cnn_model.save('pretrained_cnn_model.h5')

all_predictions300 = []
all_predictions11 = []

# random ensemble initialization
for i in range(100):
    # Use the function to make predictions
    predictions300, predictions11 = run_ensemble(sequences3, labels3, weights3, test_sequences1, comp_test_sequences1,
                                                 test_sequences2, comp_test_sequences2)
    all_predictions300.append(predictions300)
    all_predictions11.append(predictions11)

# calculate mean over the predictions of the 100 models
avg_predictions300 = np.mean(all_predictions300, axis=0)
avg_predictions11 = np.mean(all_predictions11, axis=0)

# calculate pearson correlation on validation set
corr300, _ = pearsonr(avg_predictions300, test_labels1)
corr11, _ = pearsonr(avg_predictions11, test_labels2)

print("Pearson correlation on 11 variants: ", corr11)
print("Pearson correlation on 300 variants: ", corr300)

# add columns for the 100 models average predictions and the true labels to the csv files of the validation sets
test1["Average_model_prediction"] = avg_predictions300
test2["Average_model_prediction"] = avg_predictions11
test1["True_labels"] = test_labels1
test2["True_labels"] = test_labels2

# Save the DataFrame to a CSV file
test1.to_csv('test300_with_predictions.csv', index=False)
test2.to_csv('test11_with_predictions.csv', index=False)