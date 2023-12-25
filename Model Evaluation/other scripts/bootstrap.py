import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import itertools

variant_data = pd.read_csv('carl_predictions.txt', delimiter='\t')

true_labels = np.array(variant_data.iloc[:, 0])

predicted = np.array(variant_data.iloc[:, 1])

# Get all possible combinations of vectors of size 9 from the 11 predictions vector
combinations = list(itertools.combinations(list(range(len(predicted))), 9))

correlation_values = []

for combination in combinations:
    # Create a bootstrap sample with each combination and its corresponding true labels
    bootstrap_sample = (predicted[list(combination)], true_labels[list(combination)])

    # Calculate the Pearson correlation for each combination
    correlation_values.append(pearsonr(bootstrap_sample[0], bootstrap_sample[1])[0])

# Analyze the distribution of correlation values
mean_correlation = np.mean(correlation_values)
std_deviation = np.std(correlation_values)
confidence_interval = np.percentile(correlation_values, [2.5, 97.5])

print("mean_correlation: ", mean_correlation)
print("std: ", std_deviation)
print("confidence interval: ", confidence_interval)

correlations_bootstrap = pd.DataFrame()
correlations_bootstrap["correlation"] = correlation_values
correlations_bootstrap.to_csv("Bootstrap_dBR_model_validation.csv")
