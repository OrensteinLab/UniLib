import pandas as pd
import numpy as np
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

num_of_dp = 10000
pbm = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=int(14*num_of_dp+num_of_dp*0.2), skiprows=0)
data_all = np.array(list(map(oneHotDeg, pbm['101bp sequence'])))
# data2 = np.array(list(map(oneHot, pbm['barcode'])))
# data_all = np.append(data2, data1, axis=1)
readtot_all = np.array(pbm['readtot'])
labels_all = np.array(pbm['Mean Fl'])
labels_all = labels_all/max(labels_all)



weights=np.array(pbm['readtot'])
weights = np.log(weights)
weights=weights/max(weights)
print(weights.shape)

first_data_train, data_test, first_labels_train, labels_test = train_test_split(data_all[:num_of_dp], labels_all[:num_of_dp], test_size=0.2, random_state=42)
first_weights_train, _, _, _ = train_test_split(weights[:num_of_dp], weights[:num_of_dp], test_size=0.2, random_state=42)

labels_test = np.log(labels_test)  #log!!!

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
    model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(101, 4), use_bias=True))
    #	model.add(MaxPooling1D(pool_size=100))
    #	model.add(Flatten())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    model.fit(data_train, labels_train, epochs=5, batch_size=32, verbose=1, sample_weight=weights_train)
    pred_test = model.predict(data_test)
    pred_test = np.log(pred_test)          #log!!!
    a = pearsonr(labels_test, pred_test.reshape(len(pred_test)))[0]
    print(pearsonr(labels_test, pred_test.reshape(len(pred_test))))
    pearson_corr += [a]
#	model.save("unilib_variant_mean_100_filter5000.model")



print("pearson_corr =",pearson_corr)
print("amount_of_data_points =", amount_of_data_points)
print("min_readtot =", min_readtot)

#results:

# [meanFL, no barcode, with sample weights model fit]
# pearson_corr = [0.4511221716051027, 0.4388472358804014, 0.43215852308305347, 0.42781734057500515, 0.43558375328005094, 0.4103491983428952, 0.3875244178283658, 0.36764986074477646, 0.3479813950095468, 0.31016279755172954, 0.27027497106355813, 0.2287042342539751, 0.18396920873335057, 0.15391673581233706]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [meanFL, with barcode, with sample weights model fit]
# pearson_corr = [0.4267723077715702, 0.44720297379223367, 0.4386681616618783, 0.4279744910707042, 0.43179092007028474, 0.4167435899354919, 0.394965480198018, 0.37082469766110926, 0.3355747956773678, 0.328994375831432, 0.27654980986488475, 0.2370162677820199, 0.20132151670936904, 0.11241139872346481]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [log meanFL, with barcode, with sample weights model fit]
# pearson_corr = [0.443418565895802, 0.45701605715888816, 0.4580647089605019, 0.4465357432616376, 0.4454201927998763, 0.4286502899808109, 0.42220938468353264, 0.41360985810694634, 0.3850384787806399, 0.33091185420952873, 0.33444846599012934, 0.2908330196159247, 0.25564675953164606, 0.3002268415958914]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [log meanFL, with barcode, with sample weights model fit]
# pearson_corr = [0.44976720848164947, 0.44563547539274767, 0.4466576513189845, 0.4512711202292086, 0.43633756738686524, 0.42688486953100874, 0.41089327792490515, 0.40313413850062235, 0.38795861519255015, 0.36489314486777524, 0.31427705289585417, 0.276761754018208, 0.26473503014429584, 0.24327296174774365]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [meanFL, with barcode, with sample weights  model fit] - log pearson corr
# pearson_corr = [0.45697452595023813, 0.462806398634125, 0.4569781717177102, 0.4416890560035466, 0.4321911743308702, 0.42000606590664213, 0.3920829759947355, 0.3895233798856745, 0.346541394574358, 0.2938942310328697, 0.25417714554222176, 0.21863640240227677, 0.1825572558582499, 0.16313357297392828]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]
