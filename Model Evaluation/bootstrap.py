import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import itertools

variant_data = pd.read_csv('11_validation_variants.csv')

true_labels = np.array(variant_data['yeast average'])

df=pd.read_csv('test_11_with_predictions.csv')

predicted=df['Average_model_prediction']

print(pearsonr(predicted, true_labels)[0])

# Get all possible combinations of vectors of size 9 from  predicted
combinations = list(itertools.combinations(predicted, 9))

correlation_values = []

for combination in combinations:
    # find indexes of the elements in the combination
    indexes = [np.where(predicted == element)[0][0] for element in combination]

    # Create a bootstrap sample with each combination and its corresponding true labels
    bootstrap_sample = (predicted[indexes], true_labels[indexes])

    # Calculate the Pearson correlation for each combination
    correlation_values.append(pearsonr(bootstrap_sample[0], bootstrap_sample[1])[0])

# Analyze the distribution of correlation values
mean_correlation = np.mean(correlation_values)
std_deviation = np.std(correlation_values)
confidence_interval = np.percentile(correlation_values, [2.5, 97.5])

print("mean_correlation: ", mean_correlation)
print("std: ", std_deviation)
print("confidence interval: ", confidence_interval)

bootstrap_data = pd.DataFrame()
bootstrap_data['correlation'] = correlation_values
bootstrap_data.to_csv("bootstrap_transfer_learning.csv")
