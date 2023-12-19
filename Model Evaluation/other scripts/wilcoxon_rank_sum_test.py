import pandas as pd
import scipy.stats as stats
from statsmodels.stats import multitest

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

    # dictionary which will contain mean fl values for all variants containing each motif
    motif_variants = {}

    # find all variants containing each motif
    for motif in motifs:
        # use mask to find variant indexes containing the motifs
        contains_motif_mask = all_data['101bp sequence'].str.contains(motif)
        contain_motif = all_data[contains_motif_mask]['Mean Fl']  # group of variants which contain specific motif
        motif_variants[motif] = contain_motif

    motifs.append('desert motif')
    mask_desert = all_data['variant'].apply(
        lambda seq: 'd00' in seq[3:7]
    )

    contains_desert = all_data[mask_desert]['Mean Fl']
    motif_variants['desert motif'] = contains_desert

    # sort motifs by median of mean FL
    motifs = sorted(motifs, key=lambda x: motif_variants[x].median(axis=0),
                    reverse=True)  # sort motifs by median mean fl in reverse order

    p_values = []

    for i in range(len(motifs)):
        motif = motifs[i]
        median = motif_variants[motif].median()
        print(motif, median)
        contain_motif = motif_variants[motif] # all variants containing motif
        other_motifs = all_data
        # remove all motifs in -5 or plus 5 in the ranking by median from the original motif
        for close_motif in motifs[max(0, i - 5):min(i + 6, len(motifs))]:
            other_motifs = other_motifs[~other_motifs['101bp sequence'].str.contains(close_motif)]
        statistic, p_value = stats.mannwhitneyu(contain_motif, other_motifs['Mean Fl']) # perform Wilcoxon rank sum test
        p_values.append(p_value)

    p_value_df = pd.DataFrame()

    p_value_df["motif"]=motifs

    p_value_df["Wilcoxon_rank_sum_P_value"] = p_values

    p_values_df = pd.DataFrame(p_values, index=motifs)

    # Perform Benjamini-Hochberg correction
    rejected, corrected_p_values, _, _ = multitest.multipletests(p_values_df.values.flatten(), alpha=0.1,
                                                                 method='fdr_bh')
    # Add the rejection column to the DataFrame
    p_value_df['Null hypothesis rejected'] = rejected

    p_value_df['Benjamini Hochberg corrected p value'] = corrected_p_values

    # Save the DataFrame to a CSV file
    p_value_df.to_csv('Wilcoxon_rank_sum_ranked_motifs.csv', index_label='Motif')


if __name__ == '__main__':
    main()
