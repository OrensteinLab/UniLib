import pandas as pd

def main():
    # read data into a DataFrame:
    df = pd.read_csv('unilib_variant_bindingsites_KM_mean_0_sorted.csv')

    # Save variant numbers that have more than 2 variant numbers (variants with 22 barcodes)
    variant_counts = df['variant'].value_counts()
    variant_numbers_22_barcodes = variant_counts[variant_counts > 2].index

    # 1. for each bin, calculate total number of reads
    total_reads_bin1 = df['readbin1'].sum()
    total_reads_bin2 = df['readbin2'].sum()
    total_reads_bin3 = df['readbin3'].sum()
    total_reads_bin4 = df['readbin4'].sum()

    # group all variants with the same variant number and create data frame with the sum of total reads in each bin for each variant number
    all_variants = df.groupby('variant')[['readbin1', 'readbin2', 'readbin3', 'readbin4']].sum()

    all_variants = all_variants.reset_index()

    # merge to so that the new dataframe will include the variable region column for each variant
    all_variants= all_variants.merge(df[['variant','101bp sequence']].drop_duplicates(),
                                                 on='variant')

    # 2. For each variant in a specific bin,  we divided its read count by the total number of reads of that bin to get adjusted reads
    all_variants['NormalizedBin1'] = all_variants['readbin1'] / total_reads_bin1
    all_variants['NormalizedBin2'] = all_variants['readbin2'] / total_reads_bin2
    all_variants['NormalizedBin3'] = all_variants['readbin3'] / total_reads_bin3
    all_variants['NormalizedBin4'] = all_variants['readbin4'] / total_reads_bin4

    # 3. We used the percentage area of each bin (“%Bin”)
    percentage_bin1=21.6
    percentage_bin2=21.8
    percentage_bin3=21.3
    percentage_bin4=22.6

    # 4. We multiplied the outcome in step 2 by the corresponding “%Bin”, resulting  in adjusted reads per variant per bin.
    all_variants['NormalizedBin1']= all_variants['NormalizedBin1'] * percentage_bin1
    all_variants['NormalizedBin2'] = all_variants['NormalizedBin2'] * percentage_bin2
    all_variants['NormalizedBin3'] = all_variants['NormalizedBin3'] * percentage_bin3
    all_variants['NormalizedBin4'] = all_variants['NormalizedBin4'] * percentage_bin4

    # 5. We summed the adjusted reads  in all four bins
    all_variants['sum_adjusted_reads']=all_variants['NormalizedBin1'] + all_variants['NormalizedBin2'] + all_variants['NormalizedBin3'] + all_variants['NormalizedBin4']

    # Finally, for each variant, we divided the adjusted reads per bin from  step 4 by the sum calculated in step 5, resulting in normalized reads for each bin.
    all_variants['NormalizedBin1'] = all_variants['NormalizedBin1'] / all_variants['sum_adjusted_reads']
    all_variants['NormalizedBin2'] = all_variants['NormalizedBin2'] / all_variants['sum_adjusted_reads']
    all_variants['NormalizedBin3'] = all_variants['NormalizedBin3'] / all_variants['sum_adjusted_reads']
    all_variants['NormalizedBin4'] = all_variants['NormalizedBin4'] / all_variants['sum_adjusted_reads']

    all_variants['total_reads']=all_variants['readbin1']+all_variants['readbin2']+all_variants['readbin3']+all_variants['readbin4']

    # We calculate linear combination of normalized bin values to find mean fl
    all_variants['Mean_FL'] = all_variants['NormalizedBin1'] * 607 + all_variants['NormalizedBin2'] * 1364 + all_variants['NormalizedBin3'] * 2596 + all_variants['NormalizedBin4'] * 7541

    # drop the sum_adjusted_reads column from the dataframe
    all_variants=all_variants.drop(columns=['sum_adjusted_reads'])

    all_variants.to_csv("all_aggregated_variants.csv")

    # We filter only variants with 22 barcodes
    variant_22_barcodes_df = all_variants[all_variants['variant'].isin(variant_numbers_22_barcodes)]

    # Sort the DataFrame by the 'total_reads' column in descending order
    variants_22_barcodes_sorted = variant_22_barcodes_df.sort_values(by='total_reads', ascending=False)

    # Select the top 300 rows with most reads as test set
    top_300_df = variants_22_barcodes_sorted.head(300)

    # select rest of rows as train set.
    train_variant_22_barcodes= variants_22_barcodes_sorted.tail(2435-300)

    train_variant_22_barcodes.to_csv("train_set_variants_22_barcodes.csv", index=False)

    # create file with 300 test variants
    top_300_df.to_csv("300_test_variants.csv",index=False)

    # exclude test variants from the table of all variants
    all_variants_without_test = all_variants[~all_variants['variant'].isin(top_300_df['variant'])]

    # Write data frame to csv file
    all_variants_without_test.to_csv("all_variants_without_test.csv",index=False)

if __name__ == '__main__':
    main()
