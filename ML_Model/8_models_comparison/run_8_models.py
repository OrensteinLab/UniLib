import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats.stats import pearsonr
import random
from itertools import product
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

num_of_dp = 10000
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def oneHotDeg(string):
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


def oneHot(string):
    trantab = str.maketrans('ACGT', '0123')
    string = str(string)
    data = [int(x) for x in list(string.translate(trantab))]
    return np.eye(4)[data]


def train_predict_model(model, train_seq, weights_train, mean_fl_train, bins_train, use_bins, test_data):
    if use_bins:
        model.fit(train_seq, bins_train, epochs=3, batch_size=128, verbose=1, sample_weight=weights_train) #callbacks=[lr_scheduler]
        pred_test = model.predict(test_data)
        weights_bins = [607, 1364, 2596, 7541]
        pred_meanFL = np.array([np.dot(x, weights_bins) for x in pred_test])
    else:
        model.fit(train_seq, mean_fl_train, epochs=3, batch_size=128, verbose=1, sample_weight=weights_train)
        pred_meanFL = model.predict(test_data)

    return pred_meanFL

def create_model(input_shape, use_bins):

    optimizer = Adam(learning_rate=0.0003)
    model = Sequential()
    model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=input_shape, use_bias=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16, activation='relu'))

    if use_bins:
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    else:
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizer, loss='mse')

    return model


def run_model(combination, train_data, test_data):
    
    # function to run specific model with specific combination of attributes
    use_bins, use_barcodes, use_weights = combination

    # display attribute combination
    print("bins {}, weights {}, barcode {} \n".format(use_bins, use_weights, use_barcodes))

    # read data from csv files
    all_sequences = np.array(list(map(oneHotDeg, train_data['VariableRegion'])))  # turn to one hot sequences
    all_barcodes = np.array(list(map(oneHot, train_data['UniversalAllowedBCs'])))  # read barcodes
    bins_all = np.transpose(np.array([train_data['nBin1Reads'], train_data['nBin2Reads'], train_data['nBin3Reads'],
                                      train_data['nBin4Reads']]))  # bin labels
    mean_fl_all = np.array(train_data['MeanFL'])  # read expression labels
    mean_fl_all = mean_fl_all / max(mean_fl_all)  # normalize labels
    readtot_all = np.array(train_data['TotalReads'])  # read total reads for every variant

    # read test data
    test_sequences = np.array(list(map(oneHotDeg, test_data['VariableRegion'])))
    test_barcodes = np.array(list(map(oneHot, test_data['UniversalAllowedBCs'])))
    mean_fl_test = np.array(test_data['MeanFL'])  # test labels

    # define weights
    weights = np.array(train_data['TotalReads'])
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

        cnn_model = create_model(input_shape, use_bins)  # use function to create CNN model

        print("iteration ", i)
        amount_of_data_points += [num_of_dp * (i + 1)]
        min_readtot += [readtot_all[amount_of_data_points[i]]] # find the minimum read number for the specific amount of data points
        # concatenate arrays to include 10,000 additional data points
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
        a = pearsonr(mean_fl_test, pred_meanFL.reshape(len(pred_meanFL)))[0]
        print(a)
        pearson_corr += [a]


    print("pearson_corr =", pearson_corr)
    print("amount_of_data_points =", amount_of_data_points)
    print("min_readtot =", min_readtot)

    # write all results to output file
    with open("model_compare_results.txt", 'a') as output:
        output.write("bins {}, weights {}, barcode {} \n".format(use_bins, use_weights, use_barcodes))
        output.write("reads total: {} \n".format(min_readtot))
        output.write("amount_of_data_point: {}\n".format(amount_of_data_points))
        output.write("Pearson correlation: {}\n".format(pearson_corr))


def main():
    # read csv file with 140k sequences and expression data
    all_data = pd.read_csv("T_AllRuns.csv",nrows=int(14 * num_of_dp + num_of_dp * 0.5), skiprows=0)

    all_data = all_data.sort_values(by='TotalReads', ascending=False)
    # select 4000 random indexes for top 20k in the dataframe
    random_test_indexes = random.sample(range(20000), 4000)
    # select the 4000 random rows from pbm as test set
    test_set = all_data.iloc[random_test_indexes]
    # Remove the selected rows
    train_data = all_data.drop(random_test_indexes)
    # sort dataframe again by the total reads
    train_data = train_data.sort_values(by='TotalReads', ascending=False)

    options = [True, False]
    combinations = list(product(options, repeat=3))  # create 8 combinations of 3 attributes

    # Iterate through the combinations of attributes and call the function run_model() with them
    for combination in combinations:
        run_model(combination, train_data, test_set)


if __name__ == "__main__":
    main()
