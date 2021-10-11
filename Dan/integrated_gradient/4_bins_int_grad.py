from TF_modisco import *
from scipy.io import savemat
import seaborn as sns
# force the server to run on cpu and not on Gpu
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

def oneHot(string):
    trantab = str.maketrans('ACGT', '0123')
    string = str(string)
    data = [int(x) for x in list(string.translate(trantab))]
    return np.eye(4)[data]

model = tf.keras.models.load_model("unilib_variant_20000_biggest_readtot_trained.model")
pbm = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=150000, skiprows=0)
pbm.dropna(subset = ["101bp sequence"], inplace=True)
pbm = pbm.reset_index(drop=True)
sequences_fetures = np.array(list(map(oneHotDeg, pbm["101bp sequence"])))
ex_list, hyp_ex_list = compute_impratnce_scores(model, sequences_fetures, target_range=slice(3,4,1), batch_size=32)
# run_modisco(hyp_ex_list, ex_list, sequences_fetures)
np.save("ex_list4_150000", np.array(ex_list))
