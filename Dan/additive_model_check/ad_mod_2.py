import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split

b = np.load("additive_model_dir/b.npy")
A = np.loadtxt("additive_model_dir/A.txt", dtype=int)

A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)

x = np.linalg.lstsq(A_train, b_train, rcond=None)[0]

prediction = np.zeros(b_test.shape[0], dtype=float)
for i in range(A_test.shape[0]):
    for j in range(A_test.shape[1]):
        if A_test[i][j] == 1:
            prediction[i] += x[j]


mse = np.array([((prediction - b_test)**2).mean()])
np.savetxt("additive_model_dir/mod_2_mse.txt", mse, fmt='%f')
pc = np.array([pearsonr(prediction.reshape(len(prediction)), b_test.reshape(len(prediction)))[0]])
np.savetxt("additive_model_dir/mod_2_pc.txt", pc, fmt='%f')