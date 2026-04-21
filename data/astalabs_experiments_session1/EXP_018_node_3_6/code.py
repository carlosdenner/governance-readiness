import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Define file path (one level above as per instructions)
file_path = '../step2_competency_statements.csv'

# Load dataset
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback to current directory for robustness if the file isn't found at ../
    try:
        df = pd.read_csv('step2_competency_statements.csv')
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

print(f"Dataset loaded. Shape: {df.shape}")

# Calculate control counts
# Assuming semicolon separation based on dataset description
df['control_count'] = df['applicable_controls'].apply(lambda x: len([c for c in str(x).split(';') if c.strip()]) if pd.notna(x) else 0)

# Group by bundle
bundle_groups = df.groupby('bundle')

# Descriptive Statistics
summary_stats = bundle_groups['control_count'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print("\n=== Descriptive Statistics for Control Counts ===")
print(summary_stats)

# Extract series for testing
trust_data = df[df['bundle'] == 'Trust Readiness']['control_count']
integration_data = df[df['bundle'] == 'Integration Readiness']['control_count']

# Normality Check (Shapiro-Wilk)
_, p_trust = stats.shapiro(trust_data)
_, p_int = stats.shapiro(integration_data)

print("\n=== Normality Check (Shapiro-Wilk) ===")
print(f"Trust Readiness: p={p_trust:.4f}")
print(f"Integration Readiness: p={p_int:.4f}")

# Statistical Test Selection
# Use Mann-Whitney U test if data is not normal or sample size is small, otherwise t-test
# Given n is roughly 20 per group, and count data often isn't normal, Mann-Whitney is safer.
use_parametric = (p_trust > 0.05) and (p_int > 0.05)

if use_parametric:
    test_stat, p_val = stats.ttest_ind(trust_data, integration_data, equal_var=False)
    test_name = "Welch's t-test"
else:
    test_stat, p_val = stats.mannwhitneyu(trust_data, integration_data, alternative='two-sided')
    test_name = "Mann-Whitney U test"

print(f"\n=== Hypothesis Test ({test_name}) ===")
print(f"Statistic: {test_stat:.4f}")
print(f"P-value:   {p_val:.4f}")

if p_val < 0.05:
    print("Conclusion: Significant difference in control complexity between bundles (Reject H0).")
else:
    print("Conclusion: No significant difference in control complexity (Fail to reject H0).")

# Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [trust_data, integration_data]
labels = ['Trust Readiness', 'Integration Readiness']

# Boxplot
bplot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, medianprops=dict(color='black'))

# Colors
colors = ['lightblue', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# Jitter plot overlay
for i, data in enumerate(data_to_plot):
    y = data
    x = np.random.normal(1 + i, 0.04, size=len(y))
    plt.scatter(x, y, alpha=0.6, color='darkblue', s=20)

plt.title('Complexity Comparison: Count of Architecture Controls per Competency')
plt.ylabel('Number of Applicable Controls')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()