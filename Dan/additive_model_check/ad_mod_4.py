import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from scipy.special import comb



b = np.load("additive_model_dir/b.npy")
A = np.loadtxt("additive_model_dir/A.txt", dtype=int)

new_A = np.zeros((A.shape[0],42+42**2), dtype=int)

for i in range(A.shape[0]):
    site_idx = [-1,-1,-1]
    k = 0
    for j in range(A.shape[1]):
        if A[i][j] == 1:
            new_A[i][j%42] += 1
            site_idx[k] = j % 42
            k += 1
    site_idx = sorted(site_idx)
    new_A[i][42+site_idx[0]*42+site_idx[1]] += 1
    new_A[i][42+site_idx[1]*42+site_idx[2]] += 1
    new_A[i][42+site_idx[0]*42+site_idx[2]] += 1

# #QA
# for i in range(A.shape[0]):
#     if sum(new_A[i][:]) != 6:
#         print('not great', i,sum(new_A[i][:]))
# print('great')

A = new_A

A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)

x = np.linalg.lstsq(A_train, b_train, rcond=None)[0]

prediction = np.zeros(b_test.shape[0], dtype=float)
for i in range(A_test.shape[0]):
    for j in range(A_test.shape[1]):
        if A_test[i][j] != 0:
            prediction[i] += x[j]*A_test[i][j]


mse = np.array([((prediction - b_test)**2).mean()])
np.savetxt("additive_model_dir/mod_4_mse.txt", mse, fmt='%f')
pc = np.array([pearsonr(prediction.reshape(len(prediction)), b_test.reshape(len(prediction)))[0]])
np.savetxt("additive_model_dir/mod_4_pc.txt", pc, fmt='%f')