import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('holistic_results.csv')

# Calculate the median for the chosen metric, AUPRC
median_auprc = df['AUPRC'].median()

# Bootstrapping for 95% CI of the median
n_bootstraps = 10000
bootstrapped_medians = []

for _ in range(n_bootstraps):
    sample = df['AUPRC'].sample(n=len(df['AUPRC']), replace=True)
    bootstrapped_medians.append(sample.median())

# Calculate 95% CI from bootstrapped medians
ci_lower, ci_upper = np.percentile(bootstrapped_medians, [2.5, 97.5])

print(f"Median AUPRC: {median_auprc:.4f}")
print(f"95% CI for Median AUPRC: [{ci_lower:.4f}, {ci_upper:.4f}]")


