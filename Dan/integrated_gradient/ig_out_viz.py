import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
from modisco.visualization import viz_sequence


num_of_inputs=1000

pbm = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=num_of_inputs, skiprows=0)
variants = pbm["variant"]
ex_list = np.load("ex_list1.npy")
for i in range(0, num_of_inputs):
    viz_sequence.plot_weights(ex_list[i])
    plt.savefig('int_grad_out/nbinread1/'+str(variants[i])+'.jpg')