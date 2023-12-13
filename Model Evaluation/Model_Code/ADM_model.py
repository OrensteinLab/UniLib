import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd
from scipy.stats.stats import pearsonr
num_of_dp = 10000

# Set random seeds for reproducibility
seed_value=42

os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

def oneHotDeg(string):
    """
    Convert DNA sequences to one-hot encoding with degenerate bases.

    Args:
        string (str): DNA sequence containing A, C, G, T, K, and M bases.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (101, 4).
    """
    string=str(string)
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

# read 11 validation variants
test2=pd.read_csv("11_validation_variants.csv")
test_sequence2=[seq[15:] for seq in test2['sequence']]

test_sequences2 = np.array(list(map(oneHotDeg, test_sequence2)))
mean_fl_test2 = np.array(test2['mean_fl'])  # test labels

train_data=pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv",nrows=20000,skiprows=0)

train_sequences = np.array(list(map(oneHotDeg, train_data['101bp sequence'])))  # turn to one hot sequences

mean_fl_train = np.array(train_data['Mean Fl'])  # read expression labels
mean_fl_train = mean_fl_train / max(mean_fl_train)  # normalize labels

# define weights
weights = np.array(train_data['readtot'])
weights = np.log(weights)
weights = weights / max(weights)

# create CNN model for 11 vairants validation
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101,4),
                           use_bias=True))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')


shuffled_indices = np.arange(len(train_sequences))
np.random.shuffle(shuffled_indices)
train_sequences = train_sequences[shuffled_indices]
mean_fl_train = mean_fl_train[shuffled_indices]

# fit model on train sequences
model.fit(train_sequences, mean_fl_train, epochs=5, batch_size=32, verbose=1, shuffle=True, sample_weight=weights)
pred_mean_fl2 = model.predict(np.array(test_sequences2))

# caluclate pearson correlation on 11 sURS
corr11 = pearsonr(mean_fl_test2, pred_mean_fl2.reshape(len(pred_mean_fl2)))[0]
print("Corr11: ",corr11)

test2["ADM predictions"]=pred_mean_fl2.reshape(len(pred_mean_fl2))

test2.to_csv("ADM_predictions_11_sURS.csv")

# read 300 validation variants
test1 = pd.read_csv("300_test_variants.csv")
test_sequences1 = list(test1['101bp sequence'])  # read sequences
test_sequences1 = np.array(list(map(oneHotDeg, test_sequences1)))  # use one hot function
# read labels
mean_fl_test1= test1['Mean_FL']

# read training data
train_data=pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv",nrows=20000,skiprows=0)

train_data=train_data[~train_data['101bp sequence'].isin(list(test1['101bp sequence']))] # remove 300 test sequences from train data

train_sequences = np.array(list(map(oneHotDeg, train_data['101bp sequence'])))  # turn to one hot sequences

mean_fl_train = np.array(train_data['Mean Fl'])  # read expression labels
mean_fl_train = mean_fl_train / max(mean_fl_train)  # normalize labels

# define weights
weights = np.array(train_data['readtot'])
weights = np.log(weights)
weights = weights / max(weights)


# create CNN model for 300 test validation
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101,4),
                           use_bias=True))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')


shuffled_indices = np.arange(len(train_sequences))
np.random.shuffle(shuffled_indices)
train_sequences = train_sequences[shuffled_indices]
mean_fl_train = mean_fl_train[shuffled_indices]

# fit model on train sequences
model.fit(train_sequences, mean_fl_train, epochs=5, batch_size=32, verbose=1, shuffle=True, sample_weight=weights)
pred_mean_fl1 = model.predict(np.array(test_sequences1))

# calcuclate pearson correlation on 300 test variants
corr300 = pearsonr(mean_fl_test1, pred_mean_fl1.reshape(len(pred_mean_fl1)))[0]
print("Corr11: ",corr300)


test1["ADM predictions"]=pred_mean_fl1.reshape(len(pred_mean_fl1))

test1.to_csv("ADM_predictions_300_test_variants.csv")

