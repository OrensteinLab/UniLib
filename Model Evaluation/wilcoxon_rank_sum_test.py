import pandas as pd
import scipy.stats as stats
import numpy as np


def calculate_p_values_per_motif(motif_variants,original_motif, other_motifs):
    p_values = []
    contains_original_motif = motif_variants[original_motif]

    for other_motif in other_motifs:
        contains_other_motif = motif_variants[other_motif]
        statistic, p_value = stats.mannwhitneyu(contains_original_motif, contains_other_motif)
        p_values.append(p_value)

    fisher_p_value = stats.combine_pvalues(p_values, method='fisher', weights=None)[1]
    return fisher_p_value,p_values

def main():

    all_data = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_sorted.csv")

    group1 = pd.read_csv("unilib_variant_bindingsites_KM_mean_0_pop1.csv")
    barcodes_group1 = group1['barcode']

    # filter only barcodes in the file of group 1
    all_data = all_data[all_data['barcode'].isin(barcodes_group1)]

    # filter variants with less than 12,500 reads
    all_data = all_data[all_data['readtot'] > 12500]

    motifs_table = pd.read_csv("Unilib_Motifs_info.csv")
    motifs = list(motifs_table['Motif sequence'])  # read motifs from file
    motifs.remove('GAATATTCTAGAATATTC')  # remove the motif with 0 matches in the data

    # dictionary which will contain all variants containing each motif
    motif_variants = {}

    # find all variants containing each motif
    for motif in motifs:
        # use mask to find variant indexes containing the motifs
        contains_motif_mask = all_data['101bp sequence'].str.contains(motif)
        contain_motif = all_data[contains_motif_mask]['Mean Fl'] # group of variants which contain specific motif
        motif_variants[motif] = contain_motif

    motifs.append('desert motif')
    mask_desert = all_data['variant'].apply(
        lambda seq: 'd00' in seq[3:7]
    )

    contains_desert = all_data[mask_desert]['Mean Fl']
    motif_variants['desert motif'] = contains_desert

    # sort motifs by median of mean FL
    motifs = sorted(motifs, key=lambda x: motif_variants[x].median(axis=0), reverse=True)

    p_values_matrix = np.zeros((len(motifs), len(motifs)))

    for i in range(len(motifs)):
        median=motif_variants[motifs[i]].median()
        original_motif = motifs[i]
        print(original_motif,median)
        other_motifs = motifs
        combined_p_value, p_values = calculate_p_values_per_motif(motif_variants, original_motif, other_motifs)
        p_values_matrix[i, :] = p_values

    p_values_df = pd.DataFrame(p_values_matrix, index=motifs, columns=motifs)

    # Save the DataFrame to a CSV file
    p_values_df.to_csv("motif_comparison_matrix.csv")


if __name__ == '__main__':
    main()
