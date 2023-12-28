import os
import pandas as pd
from scipy.stats.stats import pearsonr
import random
import numpy as np
import tensorflow as tf
import zipfile

os.chdir("../Datasets/")

# Set random seeds for reproducibility
seed_value = 42

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def extract_zip(zip_path):
    # Specify the destination directory (current directory in this case)
    current_directory = os.getcwd()

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all contents to the destination directory
        zip_ref.extractall(current_directory)



def oneHotDeg(string):
    """
    Convert DNA sequences to one-hot encoding with degenerate bases.

    Args:
        string (str): DNA sequence containing A, C, G, T, K, and M bases.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (101, 4).
    """
    string = str(string)
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


extract_zip('unilib_variant_bindingsites_KM_mean_0_sorted.zip')

# read 11 validation variants
test2 = pd.read_csv("11_validation_variants.csv")
test_sequence2 = [seq[15:] for seq in test2['sequence']]

test_sequences2 = np.array(list(map(oneHotDeg, test_sequence2)))
mean_fl_test2 = np.array(test2['mean_fl'])  # test labels

train_data = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=20000, skiprows=0)
# turn to one hot sequences
train_sequences = np.array(list(map(oneHotDeg, train_data['101bp sequence'])))

mean_fl_train = np.array(train_data['Mean Fl'])  # read expression labels
mean_fl_train = mean_fl_train / max(mean_fl_train)  # normalize labels

# define weights
weights = np.array(train_data['readtot'])
weights = np.log(weights)
weights = weights / max(weights)

# create CNN model for 11 vairants validation
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4),
                           use_bias=True))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')

# shuffle train data
shuffled_indices = np.arange(len(train_sequences))
np.random.shuffle(shuffled_indices)
train_sequences = train_sequences[shuffled_indices]
mean_fl_train = mean_fl_train[shuffled_indices]

# fit model on train sequences
model.fit(train_sequences, mean_fl_train, epochs=5, batch_size=32, verbose=1, shuffle=True, sample_weight=weights)
pred_mean_fl2 = model.predict(np.array(test_sequences2))

# calculate pearson correlation on 11 sURS
corr11 = pearsonr(mean_fl_test2, pred_mean_fl2.reshape(len(pred_mean_fl2)))[0]
print("Corr11: ", corr11)

# add predictions column to dataframe
test2["ADM predictions"] = pred_mean_fl2.reshape(len(pred_mean_fl2))
# save predictions for 11 validation variants as csv file
test2.to_csv("ADM_predictions_11_sURS.csv")


# read 300 validation variants
test1 = pd.read_csv("300_test_variants.csv")
test_sequences1 = list(test1['101bp sequence'])  # read sequences
test_sequences1 = np.array(list(map(oneHotDeg, test_sequences1)))  # use one hot function

# read labels
mean_fl_test1 = test1['Mean_FL']

# read training data
train_data = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=20000, skiprows=0)

# remove 300 test sequences from train data
train_data = train_data[~train_data['101bp sequence'].isin(list(test1['101bp sequence']))]
# turn to one hot sequences
train_sequences = np.array(list(map(oneHotDeg, train_data['101bp sequence'])))

mean_fl_train = np.array(train_data['Mean Fl'])  # read expression labels
mean_fl_train = mean_fl_train / max(mean_fl_train)  # normalize labels

# define weights
weights = np.array(train_data['readtot'])
weights = np.log(weights)
weights = weights / max(weights)

# create CNN model for 300 test validation
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4),
                           use_bias=True))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')

# shuffle train data
shuffled_indices = np.arange(len(train_sequences))
np.random.shuffle(shuffled_indices)
train_sequences = train_sequences[shuffled_indices]
mean_fl_train = mean_fl_train[shuffled_indices]

# fit model on train sequences
model.fit(train_sequences, mean_fl_train, epochs=5, batch_size=32, verbose=1, shuffle=True, sample_weight=weights)
pred_mean_fl1 = model.predict(np.array(test_sequences1))

# calculate pearson correlation on 300 test variants
corr300 = pearsonr(mean_fl_test1, pred_mean_fl1.reshape(len(pred_mean_fl1)))[0]
print("Corr300: ", corr300)

# add predictions column to test dataframe
test1["ADM predictions"] = pred_mean_fl1.reshape(len(pred_mean_fl1))

# save predictions for 300 test variants
test1.to_csv("ADM_predictions_300_test_variants.csv")
