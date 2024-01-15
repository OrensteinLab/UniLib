import pandas as pd
import numpy as np
from tensorflow.keras.layers import *
from scipy.stats import pearsonr
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Model
import re

seed_value = 42

os.chdir("../Datasets/")

os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

batch_size = 32

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
    one_hot_matrix = np.zeros((186, 4), dtype=np.float32)

    for i, base in enumerate(string):
        one_hot_matrix[i, :] = mapping.get(base)

    return one_hot_matrix


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

def train_test_model(train_sequences,test_sequences,motifs_present_train,motifs_present_test,labels_train):

    # initialize ML model for 186bp sequence
    sequence_input = Input(shape=(186, 4), name='sequence_input')
    conv_output = Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', use_bias=True)(sequence_input)
    global_max_pooling_output = GlobalMaxPooling1D()(conv_output)

    # Input layer for binary vector
    binary_vector_input = Input(shape=(41,), name='binary_vector_input')

    concatenated_output = Concatenate()([global_max_pooling_output, binary_vector_input])

    # Dense layer with ReLU activation
    dense_output = Dense(16, activation='relu')(concatenated_output)

    # Output layer with linear activation (for regression)
    output = Dense(1, activation='linear')(dense_output)

    # Create the model with two inputs and one output
    cnn_model = Model(inputs=[sequence_input, binary_vector_input], outputs=output)

    # Compile the model with Mean Absolute Error (MAE) loss
    cnn_model.compile(optimizer='adam', loss='mse')

    # Fit the model on train data
    cnn_model.fit([train_sequences, motifs_present_train], labels_train, epochs=5, batch_size=batch_size, verbose=1, shuffle=True)

    predictions=[pred[0] for pred in cnn_model.predict([test_sequences,motifs_present_test])]

    return predictions


def main():
    # read all motifs from table
    motifs_table = pd.read_csv("Unilib_Motifs_info.csv")
    motifs = list(motifs_table['Motif sequence'])  # read motifs from file
    motifs.remove('GAATATTCTAGAATATTC')  # remove rare motif

    # read MBO train data from csv file
    train = pd.read_csv("MBO_dataset.csv")
    train_sequences = list(train['101bp sequence'])  # read sequences
    train_sequences= list(map(custom_padding,train_sequences))
    motifs_present_train=motifs_present(train_sequences,motifs)
    train_sequences = np.array(list(map(oneHotDeg, train_sequences)))  # turn sequences to one hot vectors

    # read labels & normalize
    mean_fl_train = train['Mean_FL']
    labels_train = np.array(mean_fl_train / max(mean_fl_train))

    # read all validation sequences
    test = pd.read_csv("Yeast-data_all for MLM model.csv", nrows=42, skiprows=[11,12, 13,14,15])
    test_sequences = list(test['Sequence'])
    variant_names=list(test["Name of sequence"])

    processed_test_sequences=[]

    # apply padding for the different sequences so all sequences will have 186bp length
    for seq in test_sequences:
        if len(seq) == 161:
            seq=seq[40:141]
            seq=custom_padding(seq)
        elif len(seq)==154:
            seq=seq[27:128]
            seq=custom_padding(seq)
        else:
            seq=seq[27:213]
        processed_test_sequences.append(seq)

    motifs_present_test=motifs_present(processed_test_sequences,motifs,regex=True)

    processed_test_sequences=np.array(list(map(oneHotDeg,processed_test_sequences))) # use one hot encoding

    all_predictions=[]
    # initialize 100 ensemble models and save their predictions
    for i in range(100):
        predictions=train_test_model(train_sequences,processed_test_sequences,motifs_present_train,motifs_present_test,labels_train)
        all_predictions.append(predictions)

    # average 100 ensemble model predictions
    avg_predictions=np.mean(all_predictions,axis=0)

    pred_table = pd.DataFrame()  # create a predictions dataframe

    pred_table['Name of sequence'] = variant_names

    pred_table['ML prediction'] = avg_predictions

    # merge predictions table with the validation table
    merged_df = pd.merge(test,pred_table, on='Name of sequence',how='left')

    # Save the DataFrame to a new CSV file
    merged_df.to_csv("all_variants_predictions.csv", index=False)

    # calculate all correlations
    print("yeast-glucose ",pearsonr(list(merged_df['Yeast-Glucose']),list(merged_df['ML prediction'])))
    print("yest glycerol",pearsonr(list(merged_df['Yeast-glycerol']), list(merged_df['ML prediction'])))
    print("Yeast-glucose-39C",pearsonr(list(merged_df['Yeast-glucose-39C']), list(merged_df['ML prediction'])))
    print("Yeast-glucose - NaCl",pearsonr(list(merged_df['Yeast-glucose - NaCl']), list(merged_df['ML prediction'])))


if __name__ == "__main__":
    main()


