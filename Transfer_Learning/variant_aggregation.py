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
    filter_by_variant = df.groupby('VariantNumber')[['Bin1Reads', 'Bin2Reads', 'Bin3Reads', 'Bin4Reads']].sum()

    filter_by_variant = filter_by_variant.reset_index()

    filter_by_variant= filter_by_variant.merge(df[['VariantNumber', 'VariableRegion']].drop_duplicates(),
                                                 on='VariantNumber')

    # 2. For each variant in a specific bin,  we divided its read count by the total number of reads of that bin to get adjusted reads
    filter_by_variant['NormalizedBin1'] = filter_by_variant['Bin1Reads'] / total_reads_bin1
    filter_by_variant['NormalizedBin2'] = filter_by_variant['Bin2Reads'] / total_reads_bin2
    filter_by_variant['NormalizedBin3'] = filter_by_variant['Bin3Reads'] / total_reads_bin3
    filter_by_variant['NormalizedBin4'] = filter_by_variant['Bin4Reads'] / total_reads_bin4

    # 4. We multiplied the outcome in step 2 by the corresponding “%Bin”, resulting  in adjusted reads per variant per bin.
    filter_by_variant['NormalizedBin1']= filter_by_variant['NormalizedBin1'] * percentage_bin1
    filter_by_variant['NormalizedBin2'] = filter_by_variant['NormalizedBin2'] * percentage_bin2
    filter_by_variant['NormalizedBin3'] = filter_by_variant['NormalizedBin3'] * percentage_bin3
    filter_by_variant['NormalizedBin4'] = filter_by_variant['NormalizedBin4'] * percentage_bin4

    # 5. We summed the adjusted reads  in all four bins
    filter_by_variant['sum_adjusted_reads']=filter_by_variant['NormalizedBin1'] + filter_by_variant['NormalizedBin2'] + filter_by_variant['NormalizedBin3'] + filter_by_variant['NormalizedBin4']

    # Finally, for each variant, we divided the adjusted reads per bin from  step 4 by the sum calculated in step 5, resulting in normalized reads for each bin.
    filter_by_variant['NormalizedBin1'] = filter_by_variant['NormalizedBin1'] / filter_by_variant['sum_adjusted_reads']
    filter_by_variant['NormalizedBin2'] = filter_by_variant['NormalizedBin2'] / filter_by_variant['sum_adjusted_reads']
    filter_by_variant['NormalizedBin3'] = filter_by_variant['NormalizedBin3'] / filter_by_variant['sum_adjusted_reads']
    filter_by_variant['NormalizedBin4'] = filter_by_variant['NormalizedBin4'] / filter_by_variant['sum_adjusted_reads']

    filter_by_variant['total_reads']=filter_by_variant['Bin1Reads']+filter_by_variant['Bin2Reads']+filter_by_variant['Bin3Reads']+filter_by_variant['Bin4Reads']

    # We calculate linear combination of normalized bin values to find mean fl
    filter_by_variant['Mean_FL'] = filter_by_variant['NormalizedBin1'] * 607 + filter_by_variant['NormalizedBin2'] * 1364 + filter_by_variant['NormalizedBin3'] * 2596 + filter_by_variant['NormalizedBin4'] * 7541

    filter_by_variant=filter_by_variant.drop(columns=['sum_adjusted_reads'])

    # We filter variants with 22 barcodes
    variant_22_barcodes_df = filter_by_variant[filter_by_variant['VariantNumber'].isin(variant_numbers_22_barcodes)]

    # Sort the DataFrame by the 'total_reads' column in descending order
    variants_sorted = variant_22_barcodes_df.sort_values(by='total_reads', ascending=False)

    # Select the top 300 rows
    top_300_df = variants_sorted.head(300)

    train_variant_22_barcodes= variants_sorted.tail(2435-300)

    train_variant_22_barcodes.to_csv("train_set_variants_22_barcodes.csv", index=False)

    # create file with 300 test variants which contained the most total reads
    top_300_df.to_csv("300_test_variants.csv",index=False)

    filter_by_variant_without_test = filter_by_variant[~filter_by_variant['VariantNumber'].isin(top_300_df['VariantNumber'])]

    # Write data frame to csv file
    filter_by_variant_without_test.to_csv("all_variants_without_test.csv",index=False)

if __name__ == '__main__':
    main()
