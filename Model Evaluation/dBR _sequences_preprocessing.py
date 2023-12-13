import pandas as pd

def main():
    # Read the data from 'train_sequences.txt' using a tab delimiter and UTF-8 encoding.
    df = pd.read_csv('train_sequences.txt', delimiter='\t', encoding='utf-8', header=None)

    # Extract sequences and mean_fl values from the DataFrame.
    sequences = df.iloc[:, 0]
    mean_fl = df.iloc[:, 1]

    # Initialize lists to store filtered sequences and corresponding mean_fl values.
    new_sequences = []
    new_mean_fl = []

    # Iterate through sequences and mean_fl values.
    for seq, fl in zip(sequences, mean_fl):
        if len(seq) < 101:
            # Skip sequences with length less than 101.
            continue

        # Find the midpoint of the sequence.
        mid = len(seq) // 2

        # Extract a subsequence of length 101 centered at the midpoint.
        new_seq = seq[mid - 50:mid + 51]

        # Append the new subsequence and mean_fl value to the respective lists.
        new_sequences.append(new_seq)
        new_mean_fl.append(fl)

    # Create a new DataFrame with the filtered sequences and mean_fl values.
    new_df = pd.DataFrame({'Sequence': new_sequences, 'Mean_Fl': new_mean_fl}).reset_index(drop=True)

    # Save the new DataFrame to a CSV file named "6_million_read.csv" without including an index column.
    new_df.to_csv("6_million_reads.csv", index=False)

if __name__ == '__main__':
    main()
