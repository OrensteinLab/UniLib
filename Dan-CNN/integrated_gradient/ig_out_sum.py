import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from modisco.visualization import viz_sequence

# print DF settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def get_strt_idx(df,seq101,bs,place):
    if place == 1:
        start_idx = 14
    elif place == 2:
        start_idx = 42
    elif place == 3:
        start_idx = 69
    else:
        return False
    while seq101[start_idx:(start_idx+df['len'][bs])] != df['seq'][bs]:
        start_idx += 1
        if start_idx > 27*place:
            return False
    return start_idx


def make_df():
    d_Binding_site = {
        'b01': 'GCGTGGGAA',
        'b02': 'GTCACGTGAC',
        'b03': 'TCCTGCAGGA',
        'b04': 'GAATATTCTAGAATATTC',
        'b05': 'GCTAAGCCGC',
        'b06': 'ACGATCCTCA',
        'b07': 'GTTACCCTGC',
        'b08': 'CAGCAAAAAT',
        'b09': 'CAGGTAACAA',
        'b10': 'ATCGTACGAT',
        'b11': 'GCGCATGCGC',
        'b12': 'GGTCAAAGGTCA',
        'b13': 'TCACTCACTACGA',
        'b14': 'CGGCGCTAGC',
        'b15': 'GGTTCGAACC',
        'b16': 'GCTGCGCCAC',
        'b17': 'ACCCTTACCCT',
        'b18': 'TGCCTGAGGCA',
        'b19': 'CCCCCCGCTG',
        'b20': 'GGGGAATCCCC',
        'b21': 'CCATATATGG',
        'b22': 'TTCAAGGTCA',
        'b23': 'ATGACGKCMT',
        'b24': 'MCMCGCCCA',
        'b25': 'AMACCCACACMCC',
        'b26': 'ACCCAKACACC',
        'b27': 'GMTTACGTMAKC',
        'b28': 'ACAACAACAM',
        'b29': 'CTGACCTMCC',
        'b30': 'KMTAMGCCAC',
        'b31': 'KAGGCGCAGC',
        'b32': 'AACGAGGCKK',
        'b33': 'GMTAMGCCAC',
        'b34': 'ACCAMTCGGA',
        'b35': 'MTGTCAATCA',
        'b36': 'KGGMACACTKCCM',
        'b37': 'ACMGGAAGTG',
        'b38': 'CGCCMTGTTG',
        'b39': 'KCMGGGTAAC',
        'b40': 'TATGCAAATK',
        'b41': 'ACCMCGCCCM',
        'b42': 'MCGCCCCCTA',
    }
    for i in range(11, 53):
        if i < 20:
            d_Binding_site[i] = d_Binding_site.pop('b0' + str(i - 10))
        else:
            d_Binding_site[i] = d_Binding_site.pop('b' + str(i - 10))
    df = pd.DataFrame(d_Binding_site, index=['seq'])
    for m in ['site1_counter', 'site2_counter', 'site3_counter', 'site1_ig_tot', 'site2_ig_tot', 'site3_ig_tot','site1_ig_avg', 'site2_ig_avg', 'site3_ig_avg', 'concat_bs123']:
        df.loc[m] = 0
    df = df.transpose()
    df['len'] = [len(s) for s in df['seq']]
    for m in ['site1_ig_tot', 'site2_ig_tot', 'site3_ig_tot', 'site3_ig_tot','site1_ig_avg', 'site2_ig_avg', 'site3_ig_avg']:
        for j in range(11, 53):
            df.at[j, m] = np.zeros((df['len'][j],4))

    return df


def fill_df_counters_ig_tots(df, variants, seq101, ex_list, bin):
    for i in range(num_of_inputs):
        # skip seqs containning d00 sites
        if variants[i][0] == 'd' or variants[i][3] == 'd' or variants[i][6] == 'd':
            continue

        for j in range(1, 4):
            bs = int(variants[i][3 * j - 2:3 * j])
            df['site' + str(j) + '_counter'][bs] += 1
            strt_idx = get_strt_idx(df, seq101[i], bs, j)
            for k in range(df['len'][bs]):
                df['site' + str(j) + '_ig_tot'][bs][k] += ex_list[i][strt_idx + k]
        if i % 10000 == 0:
            df.to_pickle('ig_sum_out_bin'+bin+'/ig_out_sum_df_bin'+bin+'_'+str(i)+'.pkl')


def fill_df_ig_avgs_concatavgs(df):
    for i in range(11,53):
        for j in range(1,4):
            df['site'+str(j)+'_ig_avg'][i] = df['site'+str(j)+'_ig_tot'][i] / df['site'+str(j)+'_counter'][i]

    for i in range(11, 53):
        df.at[i, 'concat_bs123'] = np.concatenate(
            (df['site1_ig_avg'][i], np.zeros((17, 4)), df['site2_ig_avg'][i], np.zeros((17, 4)), df['site3_ig_avg'][i]),
            axis=0)


#generic
num_of_inputs = 147700
bin = '4'
#inputs
pbm = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv", nrows=num_of_inputs+1, skiprows=0)
variants = pbm["variant"]
seq101 = pbm["101bp sequence"]
ex_list = np.load("ex_list"+bin+"_150000.npy")

df = make_df()
print(df)
fill_df_counters_ig_tots(df, variants, seq101, ex_list, bin)
fill_df_ig_avgs_concatavgs(df)
print(df)
df.to_pickle('ig_out_sum_df_bin'+bin+'_final.pkl')

for i in range(11,53):
    viz_sequence.plot_weights(df['concat_bs123'][i])
    plt.savefig('ig_sum_out_bin'+bin+'/b'+str(i)+'.png')








