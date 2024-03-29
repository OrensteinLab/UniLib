import os
import pandas as pd
from scipy.stats.stats import pearsonr
from itertools import product
from keras.losses import Loss
import zipfile
import numpy as np
import random
import tensorflow as tf

os.chdir("../../Model_Evaluation/Datasets/")

# Set random seeds for reproducibility
seed_value=42

os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def extract_zip(zip_path):
    # Specify the destination directory (current directory in this case)
    current_directory = os.getcwd()

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all contents to the destination directory
        zip_ref.extractall(current_directory)


num_of_dp = 10000

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
        "M": [0.5, 0.5, 0, 0],
        "N":  [0.25, 0.25, 0.25, 0.25]
    }

    # Initialize an empty matrix with the desired shape (4x101)
    one_hot_matrix = np.zeros((101, 4), dtype=np.float32)

    for i, base in enumerate(string):
        one_hot_matrix[i, :] = mapping.get(base)

    return one_hot_matrix


def oneHot(string):
    """
    Convert DNA sequences to one-hot encoding.

    Args:
        string (str): DNA sequence containing A, C, G, T.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (4,).
    """
    trantab = str.maketrans('ACGT', '0123')
    string = str(string)
    data = [int(x) for x in list(string.translate(trantab))]
    return np.eye(4)[data]



class CustomCrossEntropyDistributionLoss(Loss):
    def __init__(self, name='custom_cross_entropy_distribution_loss', weights=None, **kwargs):
        super(CustomCrossEntropyDistributionLoss, self).__init__(name=name, **kwargs)
        self.weights = weights

    def call(self, y_true, y_pred):
        # Assuming y_true and y_pred are distributions across bins

        # Calculate the weighted cross-entropy loss
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-10) * self.weights, axis=-1)

        return tf.reduce_mean(ce_loss)


def train_predict_model(model, train_seq, weights_train, mean_fl_train, bins_train, use_bins, test_data):
    """
    Train a model and make predictions.

    Args:
        model (Sequential): Keras model.
        train_seq (np.ndarray): Training sequences.
        weights_train (np.ndarray): Training weights.
        mean_fl_train (np.ndarray): Training mean FL values.
        bins_train (np.ndarray): Training bin labels.
        use_bins (bool): Whether to use bins.
        test_data (np.ndarray): Test data.

    Returns:
        np.ndarray: Predicted mean FL values.
    """

    if use_bins:
        model.fit(train_seq, bins_train, epochs=5, batch_size=32, verbose=1, shuffle=True, sample_weight=weights_train) # fit model on train data
        pred_test = model.predict(test_data) # use model to make predictions on test data
        weights_bins = [607, 1364, 2596, 7541]
        pred_meanFL = np.array([np.dot(x, weights_bins) for x in pred_test]) # multiply bin distribution be mean FL vector
    else:
        model.fit(train_seq, mean_fl_train, epochs=5, batch_size=32, verbose=1,shuffle=True, sample_weight=weights_train)
        pred_meanFL = model.predict(test_data)

    return pred_meanFL

def create_model(input_shape, use_bins):
    """
    Create a Keras model based on the input shape and task.

    Args:
        input_shape (tuple): Input shape for the model.
        use_bins (bool): Whether to use bins.

    Returns:
        Sequential: Keras model.
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=input_shape,
                               use_bias=True))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    if use_bins:
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
        weights = tf.constant([0.13, 0.17, 0.24, 0.45], dtype=tf.float32)
        model.compile(optimizer='adam', loss=CustomCrossEntropyDistributionLoss(weights=weights))
    else:
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')

    return model


def run_model(combination,all_data,train_data, test_data):
    """
    Run the model with specific attribute combinations.

    Args:
        combination (tuple): Attribute combination (use_bins, use_barcodes, use_weights).
        train_data (pd.DataFrame): Training data.
        test_data (np.ndarray): Test data.
        all_data: all data
    """

    use_bins, use_barcodes, use_weights = combination # define combination

    # display attribute combination
    print("bins {}, weights {}, barcode {} \n".format(use_bins, use_weights, use_barcodes))

    # read data from csv files
    all_sequences = np.array(list(map(oneHotDeg, train_data['101bp sequence'])))  # turn to one hot sequences
    all_barcodes = np.array(list(map(oneHot, train_data['barcode'])))  # read barcodes
    bins_all = np.transpose(np.array([train_data['nreadBin1'], train_data['nreadBin2'], train_data['nreadBin3'],
                                      train_data['nreadBin4']]))  # bin labels
    mean_fl_all = np.array(train_data['Mean Fl'])  # read expression labels
    mean_fl_all = mean_fl_all / max(mean_fl_all)  # normalize labels
    readtot_all = np.array(all_data['readtot'])  # read total reads for every variant

    # read test data
    test_sequences = np.array(list(map(oneHotDeg, test_data['101bp sequence'])))
    test_barcodes = np.array(list(map(oneHot, test_data['barcode'])))
    mean_fl_test = np.array(test_data['Mean Fl'])  # test labels

    # define weights
    weights = np.array(train_data['readtot'])
    weights = np.log(weights)
    weights = weights / max(weights)

    # specify the steps if barcodes are used for prediction
    if use_barcodes:
        # append the barcode to the sequence
        all_data = np.append(all_barcodes, all_sequences, axis=1)
        test_data = np.append(test_barcodes, test_sequences, axis=1)
        input_shape = (116, 4)
    else:
        # if barcodes are not used for prediction, use sequence only without barcode
        all_data = all_sequences
        test_data = test_sequences
        input_shape = (101, 4)

    # take the first num_of_dp (10,000) sequences and expression values
    first_data_train = all_data[:num_of_dp]
    first_bins_train = bins_all[:num_of_dp]
    first_mean_fl_train = mean_fl_all[:num_of_dp]
    first_weights_train = weights[:num_of_dp]

    min_readtot = []
    pearson_corr = []
    amount_of_data_points = []

    for i in range(14):

        # in each iteration we add 10,000 more training example and then train and test the model
        print("iteration ", i)

        # use function to create CNN model
        cnn_model = create_model(input_shape, use_bins)

        amount_of_data_points += [num_of_dp * (i + 1)]
        min_readtot += [readtot_all[amount_of_data_points[i]]] # find the minimum amount of reads threshold

        # concatenate arrays to add 10,000 additional data points
        train_seq = np.concatenate((first_data_train, all_data[num_of_dp:int((i + 1) * num_of_dp)]), axis=0)
        bins_train = np.concatenate((first_bins_train, bins_all[num_of_dp:int((i + 1) * num_of_dp)]), axis=0)
        meanFL_train = np.concatenate((first_mean_fl_train, mean_fl_all[num_of_dp:int((i + 1) * num_of_dp)]), axis=0)

        # shuffle the indices to rearrange training array
        shuffled_indices = np.arange(len(train_seq))
        np.random.shuffle(shuffled_indices)
        train_seq = train_seq[shuffled_indices]
        bins_train = bins_train[shuffled_indices]
        meanFL_train = meanFL_train[shuffled_indices]

        # if weights are used
        if use_weights:
            # create weights_train array and shuffle it.
            weights_train = np.concatenate((first_weights_train, weights[num_of_dp:int((i + 1) * num_of_dp)]), axis=0)
            weights_train = weights_train[shuffled_indices]
        else:
            weights_train = None

        # train model and make predictions on expression values using function
        pred_meanFL = train_predict_model(cnn_model, train_seq, weights_train, meanFL_train, bins_train, use_bins, test_data)

        # calculate pearson correlation between predicted and true mean_fl
        r = pearsonr(mean_fl_test, pred_meanFL.reshape(len(pred_meanFL)))[0]
        print(r)
        pearson_corr += [r]

    print("pearson_corr =", pearson_corr)
    print("amount_of_data_points =", amount_of_data_points)
    print("min_readtot =", min_readtot)

    # write all results to output file
    with open("model_compare_results_final.txt", 'a') as output:
        output.write("bins {}, weights {}, barcode {} \n".format(use_bins, use_weights, use_barcodes))
        output.write("reads total: {} \n".format(min_readtot))
        output.write("amount_of_data_point: {}\n".format(amount_of_data_points))
        output.write("Pearson correlation: {}\n".format(pearson_corr))


def main():
    # Extract zip file
    extract_zip('unilib_variant_bindingsites_KM_mean_0_sorted.zip')

    # read csv file with 140k sequences and expression data
    all_data = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv")

    # Select 2,000 random rows for the test set
    test_set = all_data.head(10000).sample(n=2000, random_state=42)

    # Remove the selected rows from the training data
    train_data = all_data.drop(test_set.index)

    # Sort training data again by 'TotalReads'
    train_data = train_data.sort_values(by='readtot', ascending=False)

    train_data = train_data.reset_index(drop=True)

    options = [True, False]
    combinations = list(product(options, repeat=3))  # create 8 combinations of 3 attributes

    # Iterate through the combinations of attributes and call the function run_model() with them
    for combination in combinations:
        run_model(combination,all_data, train_data, test_set)


if __name__ == "__main__":
    main()
