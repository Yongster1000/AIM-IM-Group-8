import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('Holistic_results.csv')

# Calculate the mean and std for a chosen metric, for example AUPRC
mean_auprc = df['AUPRC'].mean()
std_auprc = df['AUPRC'].std()
n = len(df['AUPRC'])

# 95% CI for AUPRC
z_value = 1.96
ci_lower = mean_auprc - z_value * (std_auprc / np.sqrt(n))
ci_upper = mean_auprc + z_value * (std_auprc / np.sqrt(n))

print(f"Mean AUPRC: {mean_auprc:.4f}")
print(f"95% CI for AUPRC: [{ci_lower:.4f}, {ci_upper:.4f}]")

