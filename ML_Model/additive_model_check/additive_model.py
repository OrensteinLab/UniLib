import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import tensorflow as tf
from scipy.sparse.linalg import dsolve

#------------------------- produce A,B,x such Ax=b ----------------------------------

# # force the server to run on cpu and not on Gpu
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

def get_site_idx(var):
    x = []
    for j in range(1, 4):
        x.append(int(var[3 * j - 2:3 * j])-11)
    return tuple(x)

def make_A_and_inputs(A, inputs, variants, seqs):
    inputs_idx = 0
    combs = np.zeros((42,42,42),dtype=int)
    for i in range(147700):
        if variants[i][0] == 'd' or variants[i][3] == 'd' or variants[i][6] == 'd':
            continue
        comb_idx = get_site_idx(variants[i])
        if combs[comb_idx] == 0:
            combs[comb_idx] = 1
            inputs[inputs_idx] = seqs[i]
            A_row = np.zeros((126,))
            for j in range(3):
                A_row[comb_idx[j]+42*j] = 1
            A[inputs_idx,:] = A_row
            inputs_idx += 1


inputs = np.empty((65389,), dtype='<U101')
A = np.zeros((65389,126), dtype=int)

pbm = pd.read_csv("/data/danbenam/pycharm2/integrated_gradient/unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=147700)
variants = pbm["variant"]
seqs = pbm['101bp sequence']
normal_coef = max(pbm['Mean Fl'][:20000])

make_A_and_inputs(A, inputs, variants, seqs)

inputs = np.array(list(map(oneHotDeg,inputs)))
np.savetxt("additive_model_dir/A.txt", A, fmt='%d')
np.save("additive_model_dir/inputs.npy", inputs)

model = tf.keras.models.load_model("unilib_variant_20000_biggest_readtot_trained_meanFL.model")
b = model.predict(inputs) * normal_coef
np.save("additive_model_dir/b.npy", b)

# x = np.linalg.lstsq(A,b,rcond=None)[0]
# np.save("additive_model_dir/x.npy", x)
# print(x)
#  ------------------------ save as DF ------------------------------------

# # print DF settings
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
#
# x = np.load("additive_model_dir/x.npy")
# for i in range(len(x)):
#     if x[i] < 1:
#         x[i] = 0
# df = pd.read_pickle("/data/danbenam/pycharm2/integrated_gradient/ig_out_sum_df_bin1_final.pkl")
# x = x.flatten()
# print(x)
# df.drop(df.columns.difference(['seq']), 1, inplace=True)
# df.insert(loc=0, column="binding site", value=['b'+str(i) for i in list(df.index.values)])
# print(df)
# for i in range(3):
#     df["meanFL in place "+str(i+1)] = x[i*42:i*42+42]
# print(df)
#
# df.to_csv("additive_model_dir/additive_res.csv", index = False)

#  ------------------------ check things on results ------------------------------------
# b = np.load("additive_model_dir/b.npy")
# df = pd.read_csv("additive_model_dir/additive_res.csv")
# m1 = np.mean(df["meanFL in place 1"])
# m2 = np.mean(df["meanFL in place 2"])
# m3 = np.mean(df["meanFL in place 3"])
# m = np.mean(b)
# print(m1+m2+m3,'\n', m1, '\n',m2,'\n', m3,'\n', m)