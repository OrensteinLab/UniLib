import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats.stats import pearsonr
import tensorflow as tf
from sklearn.model_selection import train_test_split


def oneHotDeg(string):
    current_data = np.array([])
    for d in string:
        if d == "A":
            current_data = np.append(current_data, np.array([1, 0, 0, 0]).T)
        if d == "C":
            current_data = np.append(current_data, np.array([0, 1, 0, 0]).T)
        if d == "G":
            current_data = np.append(current_data, np.array([0, 0, 1, 0]).T)
        if d == "T":
            current_data = np.append(current_data, np.array([0, 0, 0, 1]).T)
        if d == "K":
            current_data = np.append(current_data, np.array([0, 0, 0.5, 0.5]).T)
        if d == "M":
            current_data = np.append(current_data, np.array([0.5, 0.5, 0, 0]).T)
    return current_data.reshape(int(len(current_data) / 4), 4)


def oneHot(string):
    trantab = str.maketrans('ACGT', '0123')
    string = str(string)
    data = [int(x) for x in list(string.translate(trantab))]
    return np.eye(4)[data]

num_of_dp = 100
pbm = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=int(14*num_of_dp+num_of_dp*0.2), skiprows=0)
data1= np.array(list(map(oneHotDeg, pbm['101bp sequence'])))
data2 = np.array(list(map(oneHot, pbm['barcode'])))
data_all = np.append(data2, data1, axis=1)
readtot_all = np.array(pbm['readtot'])
labels_all = np.transpose(np.array([pbm['nreadBin1'], pbm['nreadBin2'], pbm['nreadBin3'], pbm['nreadBin4']]))
meanFL_all = np.array(pbm['Mean Fl'])
first_data_train, data_test, first_labels_train, labels_test = train_test_split(data_all[:num_of_dp], labels_all[:num_of_dp], test_size=0.2, random_state=42)
_, _, _, meanFL_test = train_test_split(data_all[:num_of_dp], meanFL_all[:num_of_dp], test_size=0.2, random_state=42)
meanFL_test = np.log(meanFL_test)     #log!!!
min_readtot = []
k_fold_pearson_std = []
pearson_corr = []
amount_of_data_points = []


for i in range(14):
    print("iteration ", i)
    amount_of_data_points += [num_of_dp * (i+1)]
    min_readtot += [readtot_all[amount_of_data_points[i]]]
    data_train = np.concatenate((first_data_train, data_all[num_of_dp:int((i+1)*num_of_dp+num_of_dp*0.2)]), axis=0)
    labels_train = np.concatenate((first_labels_train, labels_all[num_of_dp:int((i+1)*num_of_dp+num_of_dp*0.2)]), axis=0)
    #	weights=np.array(pbm[6])
    #	weights=weights/max(weights)
    #	print(weights.shape)

    perm = np.random.permutation(len(data_train))
    labels_train = labels_train[perm]
    data_train = data_train[perm]

    model = Sequential()
    model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(116, 4), use_bias=True))
    #	model.add(MaxPooling1D(pool_size=100))
    #	model.add(Flatten())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.fit(data_train, labels_train, epochs=5, batch_size=32, verbose=1)
    pred_test = model.predict(data_test)
    wheights_bins = [607, 1364, 2596, 7541]
    pred_meanFL = np.log(np.array([np.dot(x, wheights_bins) for x in pred_test]))        #log!!!
    a = pearsonr(meanFL_test, pred_meanFL.reshape(len(pred_test)))[0]
    print(pearsonr(meanFL_test, pred_meanFL.reshape(len(pred_test))))
    pearson_corr += [a]
#	model.save("unilib_variant_mean_100_filter5000.model")



print("pearson_corr =",pearson_corr)
print("amount_of_data_points =", amount_of_data_points)
print("min_readtot =", min_readtot)

#results:

# [4 bins, no barcode, no sample weights]
# pearson_corr = [0.43921874232785374, 0.44286284421504263, nan, nan, 0.40805303410145455, 0.4028971010679241, 0.3928478565275008, 0.3667106560825276, 0.37656429269766284, 0.3346718376580219, 0.31938213325102105, 0.24621832228379834, 0.21387377643257244, 0.22015506020283265]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [4 bins, with barcode, no sample weights]
# pearson_corr = [0.4439650800010919, 0.42151971724646586, 0.4270775971861523, nan, nan, 0.4054778192463313, 0.3821858148804287, 0.37474176566201445, 0.36925514637678336, 0.34208641415609675, 0.28990041903329605, 0.24643574979200106, 0.18329922346339117, 0.16140957280868912]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [log pearson cor, 4 bins, with barcode, no sample weights]
# pearson_corr = [0.4358032093151015, 0.4547533322705739, nan, nan, 0.40829684504964675, 0.41011862723353865, 0.3743890729896613, 0.3642197544685713, 0.35731018727019, 0.3282345423955814, 0.3094245526090286, 0.20799973688410134, 0.11931043808205313, 0.1482309109156145]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]


