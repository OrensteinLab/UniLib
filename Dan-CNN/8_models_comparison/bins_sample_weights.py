import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats.stats import pearsonr
from scipy.stats.stats import tstd

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
data1 = np.array(list(map(oneHotDeg, pbm['101bp sequence'])))
data2 = np.array(list(map(oneHot, pbm['barcode'])))
data_all = np.append(data2, data1, axis=1)
readtot_all = np.array(pbm['readtot'])
labels_all = np.transpose(np.array([pbm['nreadBin1'], pbm['nreadBin2'], pbm['nreadBin3'], pbm['nreadBin4']]))
meanFL_all = np.array(pbm['Mean Fl'])

weights=np.array(pbm['readtot'])
weights = np.log(weights)
weights=weights/max(weights)
print(weights.shape)
first_data_train, data_test, first_labels_train, labels_test = train_test_split(data_all[:num_of_dp], labels_all[:num_of_dp], test_size=0.2, random_state=42)
first_weights_train, _, _, meanFL_test = train_test_split(weights[:num_of_dp], meanFL_all[:num_of_dp], test_size=0.2, random_state=42)
meanFL_test = np.log(meanFL_test)      #log!!!
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
    weights_train = np.concatenate((first_weights_train, weights[num_of_dp:int((i+1)*num_of_dp+num_of_dp*0.2)]), axis=0)


    perm = np.random.permutation(len(data_train))
    labels_train = labels_train[perm]
    data_train = data_train[perm]
    weights_train= weights_train[perm]


    model = Sequential()
    model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(116, 4), use_bias=True))
    #	model.add(MaxPooling1D(pool_size=100))
    #	model.add(Flatten())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.fit(data_train, labels_train, epochs=5, batch_size=32, verbose=1, sample_weight=weights_train)
    pred_test = model.predict(data_test)
    wheights_bins = [607, 1364, 2596, 7541]
    pred_meanFL = np.log(np.array([np.dot(x, wheights_bins) for x in pred_test]))    #log!!!
    print(tstd(pred_meanFL), tstd(meanFL_test))
    a = pearsonr(meanFL_test, pred_meanFL.reshape(len(pred_test)))[0]
    print(pearsonr(meanFL_test, pred_meanFL.reshape(len(pred_test))))
    pearson_corr += [a]
#	model.save("unilib_variant_mean_100_filter5000.model")



print("pearson_corr =",pearson_corr)
print("amount_of_data_points =", amount_of_data_points)
print("min_readtot =", min_readtot)

#results:

# [4 bins, no barcode, with log sample weights model fit]
# pearson_corr = [0.44114952811908825, 0.4463873054110701, 0.43636829096468244, nan, nan, 0.3999228411313246, 0.3959662816325887, 0.3745770781028663, 0.3666363874027931, 0.34452796302001754, 0.32229018732723175, 0.27135947439494834, 0.215393583806063, 0.2942022716583792]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]
#
# [4 bins, with barcode, with log sample weights model fit]
# pearson_corr = [0.42813257612426037, 0.4397767222770199, 0.435341918421407, nan, 4227549512323934, 0.4103730333494126, 0.382280451050999, 0.3900729134358011, 0.36408796462362214, 0.3327079899596418, 0.32322293326327267, 0.2737769977104497, 0.26264802883053834, 0.22870101160384462]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [log pearson cor, 4 bins, with barcode, with log sample weights model fit]
# pearson_corr = [0.43900076716425607, 0.44643746981647603, 0.44899249113909984, nan, 0.40661168429301264, 0.400486784086635, nan, 0.36793402703704414, 0.36135754086400795, 0.3287757692894872, 0.31264886379707846, 0.254120013233437, 0.1953262740307708, 0.11437459442421824]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [nan, 0.4130106166268076, 0.4035500308277098, 0.37555677068808335]

