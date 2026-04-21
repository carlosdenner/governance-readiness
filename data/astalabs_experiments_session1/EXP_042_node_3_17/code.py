import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Define file path based on instructions
file_name = 'step2_crosswalk_matrix.csv'
# Try loading from parent directory first as per instruction, fall back to current if not found
if os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    file_path = file_name

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File {file_name} not found in ../ or current directory.")
    exit(1)

# Validate structure
# Columns 0-5 are metadata, Columns 6+ are architecture controls
metadata_cols = df.columns[:6]
control_cols = df.columns[6:]

print(f"Metadata columns ({len(metadata_cols)}): {list(metadata_cols)}")
print(f"Control columns ({len(control_cols)}): {list(control_cols)}")

# Calculate 'control_count': sum of non-null/non-empty cells in control columns
# We check for not null and not empty string just in case
df['control_count'] = df[control_cols].apply(lambda x: x.notna() & (x != '')).sum(axis=1)

# Group by Bundle
bundles = df['bundle'].unique()
print(f"\nBundles found: {bundles}")

trust_data = df[df['bundle'] == 'Trust Readiness']['control_count']
integration_data = df[df['bundle'] == 'Integration Readiness']['control_count']

# Descriptive Statistics
print("\n--- Descriptive Statistics for Control Counts ---")
print(f"Trust Readiness (n={len(trust_data)}): Mean={trust_data.mean():.2f}, Std={trust_data.std():.2f}")
print(f"Integration Readiness (n={len(integration_data)}): Mean={integration_data.mean():.2f}, Std={integration_data.std():.2f}")

# Independent Samples T-Test
# We use Welch's t-test (equal_var=False) as sample sizes and variances might differ
t_stat, p_val = stats.ttest_ind(trust_data, integration_data, equal_var=False)

print("\n--- T-Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")

# Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [trust_data, integration_data]
plt.boxplot(data_to_plot, labels=['Trust Readiness', 'Integration Readiness'], patch_artist=True)
plt.title('Distribution of Architecture Control Counts per Requirement')
plt.ylabel('Number of Mapped Controls')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()