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
data1 = np.array(list(map(oneHotDeg, pbm['101bp sequence'])))
data2 = np.array(list(map(oneHot, pbm['barcode'])))
data_all = np.append(data2, data1, axis=1)
readtot_all = np.array(pbm['readtot'])
labels_all = np.array(pbm['Mean Fl'])
labels_all = labels_all/max(labels_all)
first_data_train, data_test, first_labels_train, labels_test = train_test_split(data_all[:num_of_dp], labels_all[:num_of_dp], test_size=0.2, random_state=42)
min_readtot = []
pearson_corr = []
amount_of_data_points = []
labels_test = np.log(labels_test)  #log!!!

for i in range(14):
    print("iteration ", i)
    amount_of_data_points += [num_of_dp * (i+1)]
    min_readtot += [readtot_all[amount_of_data_points[i]]]
    data_train = np.concatenate((first_data_train, data_all[num_of_dp:int((i+1)*num_of_dp+num_of_dp*0.2)]), axis=0)
    labels_train = np.concatenate((first_labels_train, labels_all[num_of_dp:int((i+1)*num_of_dp+num_of_dp*0.2)]), axis=0)

    perm = np.random.permutation(len(data_train))
    labels_train = labels_train[perm]
    data_train = data_train[perm]

    model = Sequential()
    model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(116, 4), use_bias=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    model.fit(data_train, labels_train, epochs=5, batch_size=32, verbose=1)
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

# [meanFL, no barcode, no sample weights]
# pearson_corr = [0.4430852154160887, 0.4509694804524299, 0.45093567791092903, 0.4433060538432929, 0.41084638771737847, 0.4208453726868934, 0.39647527102645336, 0.3689539757657422, 0.3412701251899021, 0.326218807640125, 0.2950895416412297, 0.2157644049262403, 0.15034433122292876, 0.15108644222751635]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [meanFL, with barcode, no sample weights]
# pearson_corr = [0.42545377889645447, 0.4517120992311476, 0.44790163709721587, 0.4188575828912293, 0.43576296680173393, 0.4214585842520622, 0.4008469785153387, 0.3731506290772244, 0.3498895013276486, 0.3054362210299629, 0.279929925859923, 0.2002524397410897, 0.16280524523649853, 0.14385669607565915]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]


# [log label meanFL, with barcode, no sample weights]
# pearson_corr = [0.4488152617954766, 0.45300675089171605, 0.45282120641826495, 0.4251091110823297, 0.4445411538930527, 0.4328226828202526, 0.4148957506091534, 0.3872302138329518, 0.35677994667446666, 0.34938272474808474, 0.32369349677406123, 0.2697107896872501, 0.20997866572013785, 0.2110503160424364]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]


# [log label meanFL, with barcode, no sample weights]
# pearson_corr = [0.4464251680880913, 0.45430519000906466, 0.4445241227530534, 0.4349240630850823, 0.4285239045606932, 0.41821794839592863, 0.40009247962748895, 0.3951762627943167, 0.3908454249553459, 0.3662041611710541, 0.34727503938693394, 0.2777043366493934, 0.20318487406304708, 0.2111798719094365]
# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]

# [meanFL, with barcode, no sample weights] - log pearson corr
# pearson_corr = [0.4550943879888908, 0.4670954896206748, 0.46011826725755156, 0.44373577038085377, 0.42153268289728946, 0.4216382047080302, 0.3969782710552274, 0.36705668990984086, 0.3488370353725618, 0.28592078023700424, 0.24337765514927698, 0.17128629637434636, 0.1110269265444366, 0.10155213331517621]# amount_of_data_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
# min_readtot = [8957, 6009, 4447, 3418, 2671, 2084, 1623, 1245, 934, 670, 454, 271, 119, 4]
