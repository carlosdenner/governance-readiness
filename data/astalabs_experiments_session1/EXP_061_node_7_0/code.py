import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys
import numpy as np

# Robust file loading
filename = 'step3_incident_coding.csv'
file_path = None

if os.path.exists(filename):
    file_path = filename
elif os.path.exists(os.path.join('..', filename)):
    file_path = os.path.join('..', filename)
else:
    print(f"Error: {filename} not found in current or parent directory.")
    sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Data Preparation
# Group: Security (harm_type == 'security') vs Other
security_counts = df[df['harm_type'] == 'security']['technique_count']
other_counts = df[df['harm_type'] != 'security']['technique_count']

# Descriptive Statistics
print("\n=== Descriptive Statistics (Technique Count) ===")
print(f"Security Incidents (n={len(security_counts)}):")
print(f"  Mean: {security_counts.mean():.2f}")
print(f"  Median: {security_counts.median():.2f}")
print(f"  Std Dev: {security_counts.std():.2f}")

print(f"\nOther Incidents (n={len(other_counts)}):")
print(f"  Mean: {other_counts.mean():.2f}")
print(f"  Median: {other_counts.median():.2f}")
print(f"  Std Dev: {other_counts.std():.2f}")

# Statistical Analysis: Mann-Whitney U Test
# Testing if Security > Other
u_stat, p_val = stats.mannwhitneyu(security_counts, other_counts, alternative='greater')

print("\n=== Mann-Whitney U Test Results ===")
print(f"U-statistic: {u_stat}")
print(f"P-value (one-sided, Security > Other): {p_val:.5f}")

if p_val < 0.05:
    print("Result: Statistically significant. Security incidents involve significantly more techniques.")
else:
    print("Result: Not statistically significant.")

# Visualization: Boxplot with Strip Plot overlay
plt.figure(figsize=(10, 6))
data = [security_counts, other_counts]
labels = [f'Security (n={len(security_counts)})', f'Other (n={len(other_counts)})']

# Boxplot
box = plt.boxplot(data, patch_artist=True, labels=labels, widths=0.6)

# Styling
colors = ['#ff9999', '#99ccff']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add individual data points (jittered)
for i, d in enumerate(data):
    y = d
    x = np.random.normal(1 + i, 0.04, size=len(y))
    plt.scatter(x, y, alpha=0.6, color='black', s=20, zorder=3)

plt.title('Complexity of Attack Chains: Security vs. Other Harm Types')
plt.ylabel('Technique Count (Kill Chain Length)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()