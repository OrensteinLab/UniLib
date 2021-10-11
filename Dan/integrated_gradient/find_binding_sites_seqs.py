import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pbm = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=1000, skiprows=0)
variants = pbm["variant"]
seqs = pbm["101bp sequence"]

d = {'binding_site_b':[0]+[i for i in range(11,53)], 'nucleotides': None}
df = pd.DataFrame(d)
print(df['nucleotides'])
i=0
print(variants[i][4:6])
while not all(df['nucleotides']):
    if variants[i][0] == 'd' or variants[i][3] == 'd' or variants[i][6] == 'd':
        continue
    bs1 = int(variants[i][1:3])
    if not df['nucleotides'][df.binding_site_b == bs1]:
        df['nucleotides'][df.binding_site_b == bs1] = pbm["101bp sequence"][17:28]
    bs2 = int(variants[i][4:6])
    if not df['nucleotides'][df.binding_site_b == bs2]:
        df['nucleotides'][df.binding_site_b == bs2] = pbm["101bp sequence"][45:56]
    bs3 = int(variants[i][7:9])
    if not df['nucleotides'][df.binding_site_b == bs3]:
        df['nucleotides'][df.binding_site_b == bs3] = pbm["101bp sequence"][73:84]
    i += 1

print(df)
