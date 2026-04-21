import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
# Loading from one level above as per instructions
try:
    df = pd.read_csv('../step2_crosswalk_matrix.csv')
except FileNotFoundError:
    df = pd.read_csv('step2_crosswalk_matrix.csv')

# Identify control columns (excluding metadata)
# Metadata columns are the first 6 columns
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [c for c in df.columns if c not in metadata_cols]

# Calculate control count per row (robust against case/whitespace)
# We count cells containing 'X'
def count_controls(row):
    count = 0
    for val in row:
        if isinstance(val, str) and val.strip().upper() == 'X':
            count += 1
    return count

df['control_count'] = df[control_cols].apply(count_controls, axis=1)

# Group by bundle
trust_scores = df[df['bundle'] == 'Trust Readiness']['control_count']
integration_scores = df[df['bundle'] == 'Integration Readiness']['control_count']

# Statistics
print("=== Descriptive Statistics: Control Density by Bundle ===")
desc = df.groupby('bundle')['control_count'].describe()
print(desc)

# T-Test (Welch's for unequal variances)
t_stat, p_val = stats.ttest_ind(trust_scores, integration_scores, equal_var=False)

print("\n=== T-Test Results ===")
print(f"Trust Readiness Mean: {trust_scores.mean():.2f} (n={len(trust_scores)})")
print(f"Integration Readiness Mean: {integration_scores.mean():.2f} (n={len(integration_scores)})")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print("Result: Significant difference detected (p < 0.05).")
else:
    print("Result: No significant difference detected (p >= 0.05).")

# Histogram
plt.figure(figsize=(10, 6))
# Calculate common bins for clean alignment
min_val = df['control_count'].min()
max_val = df['control_count'].max()
bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

plt.hist(trust_scores, bins=bins, alpha=0.6, label='Trust Readiness', density=True, color='skyblue', edgecolor='black')
plt.hist(integration_scores, bins=bins, alpha=0.6, label='Integration Readiness', density=True, color='orange', edgecolor='black')

plt.title('Distribution of Architecture Control Density by Readiness Bundle')
plt.xlabel('Number of Controls Mapped')
plt.ylabel('Density')
plt.legend()
plt.xticks(np.arange(min_val, max_val + 1, 1))
plt.grid(axis='y', alpha=0.3)
plt.show()