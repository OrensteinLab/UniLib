import pandas as pd

def main():
    # read data into a DataFrame:
    df = pd.read_csv('T_AllRuns.csv')

    # Save variant numbers that have more than 2 variant numbers (i.e variants with 22 barcodes)
    variant_counts = df['VariantNumber'].value_counts()
    variant_numbers_22_barcodes = variant_counts[variant_counts > 2].index

    # 1. for each bin, calculate total number of reads
    total_reads_bin1 = df['Bin1Reads'].sum()
    total_reads_bin2 = df['Bin2Reads'].sum()
    total_reads_bin3 = df['Bin3Reads'].sum()
    total_reads_bin4 = df['Bin4Reads'].sum()


    # 3. We used the percentage area of each bin (“%Bin”)
    percentage_bin1=21.6
    percentage_bin2=21.8
    percentage_bin3=21.3
    percentage_bin4=22.6


    # group all variants with the same variant number and create data frame with the sum of total reads in each bin for each variant number
    all_variants = df.groupby('VariantNumber')[['Bin1Reads', 'Bin2Reads', 'Bin3Reads', 'Bin4Reads']].sum()

    all_variants = all_variants.reset_index()

    all_variants= all_variants.merge(df[['VariantNumber', 'VariableRegion']].drop_duplicates(),
                                                 on='VariantNumber')

    # 2. For each variant in a specific bin,  we divided its read count by the total number of reads of that bin to get adjusted reads
    all_variants['NormalizedBin1'] = all_variants['Bin1Reads'] / total_reads_bin1
    all_variants['NormalizedBin2'] = all_variants['Bin2Reads'] / total_reads_bin2
    all_variants['NormalizedBin3'] = all_variants['Bin3Reads'] / total_reads_bin3
    all_variants['NormalizedBin4'] = all_variants['Bin4Reads'] / total_reads_bin4

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

    all_variants['total_reads']=all_variants['Bin1Reads']+all_variants['Bin2Reads']+all_variants['Bin3Reads']+all_variants['Bin4Reads']

    # We calculate linear combination of normalized bin values to find mean fl
    all_variants['Mean_FL'] = all_variants['NormalizedBin1'] * 607 + all_variants['NormalizedBin2'] * 1364 + all_variants['NormalizedBin3'] * 2596 + all_variants['NormalizedBin4'] * 7541

    all_variants=all_variants.drop(columns=['sum_adjusted_reads'])

    # We filter variants with 22 barcodes
    variant_22_barcodes_df = all_variants[all_variants['VariantNumber'].isin(variant_numbers_22_barcodes)]

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
    all_variants_without_test = all_variants[~all_variants['VariantNumber'].isin(top_300_df['VariantNumber'])]

    # Write data frame to csv file
    all_variants_without_test.to_csv("all_variants_without_test.csv",index=False)

if __name__ == '__main__':
    main()