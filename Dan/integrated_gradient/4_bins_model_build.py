import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from scipy.stats.stats import pearsonr
import tensorflow as tf

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

pbm = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=20000, skiprows=0)

# with barcode:
# data1 = np.array(list(map(oneHotDeg, pbm['101bp sequence'])))
# data2 = np.array(list(map(oneHot, pbm['barcode'])))
# data = np.append(data2, data1, axis=1)
#without barcode:
data = np.array(list(map(oneHotDeg, pbm['101bp sequence'])))
#
readtot = np.array(pbm['readtot'])
meanFL = np.array(pbm['Mean Fl'])
labels = np.transpose(np.array([pbm['nreadBin1'], pbm['nreadBin2'], pbm['nreadBin3'], pbm['nreadBin4']]))

perm = np.random.permutation(len(data))
labels = labels[perm]
data = data[perm]
meanFL = meanFL[perm]


model = Sequential()
model.add(Conv1D(filters=1024, kernel_size=6, strides=1, activation='relu', input_shape=(116, 4), use_bias=True))
#	model.add(MaxPooling1D(pool_size=100))
#	model.add(Flatten())
model.add(GlobalMaxPooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

# weights_train = np.append(weights[1:i],weights[1+i+fold:],axis=0)

model.fit(data, labels, epochs=5, batch_size=64, verbose=1)
# pred_test = model.predict(data_test)
# wheights = [607,1364,2596,7541]
# pred_meanFL = np.array([np.dot(x,wheights) for x in pred_test])
# a = pearsonr(meanFL_test, pred_meanFL.reshape(len(pred_test)))[0]
# print(pearsonr(meanFL_test, pred_meanFL.reshape(len(pred_test))))
# pearson += [a]
model.save("unilib_variant_20000_biggest_readtot_trained.model")

#	print(sumpearson/k)
# pearson_corr += [sum(pearson) / k]
# k_fold_pearson_std += [np.std(pearson)]

