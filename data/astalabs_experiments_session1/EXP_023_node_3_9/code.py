import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# [debug] Check current directory and parent directory contents if needed, but per instructions assume ../ exists
# Load dataset
filename = 'step2_competency_statements.csv'
filepath = f'../{filename}'

# Fallback if running in an environment where file is in current dir
if not os.path.exists(filepath):
    if os.path.exists(filename):
        filepath = filename
    else:
        # Start debug print
        print(f"File not found at {filepath} or {filename}")
        # End debug print

print(f"Loading {filepath}...")
try:
    df = pd.read_csv(filepath)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Parse 'applicable_controls' to count number of controls per statement
# Format is semicolon separated string
def count_controls(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Split by semicolon and filter empty strings
    return len([x for x in str(val).split(';') if x.strip()])

df['control_count'] = df['applicable_controls'].apply(count_controls)

# Group data by bundle
trust_data = df[df['bundle'] == 'Trust Readiness']['control_count']
integration_data = df[df['bundle'] == 'Integration Readiness']['control_count']

# Calculate statistics
stats_summary = df.groupby('bundle')['control_count'].agg(['mean', 'median', 'std', 'count', 'sem'])
print("\n=== Architectural Density Statistics (Controls per Statement) ===")
print(stats_summary)

# Perform Statistical Test
# Shapiro-Wilk test for normality
_, p_trust_norm = stats.shapiro(trust_data)
_, p_int_norm = stats.shapiro(integration_data)

print(f"\nNormality Check (Shapiro-Wilk): Trust (p={p_trust_norm:.4f}), Integration (p={p_int_norm:.4f})")

# Use Mann-Whitney U Test (Non-parametric) as count data is often non-normal
u_stat, p_val_mw = stats.mannwhitneyu(trust_data, integration_data, alternative='two-sided')

# Use Welch's T-test (for comparison, robust to unequal variances)
t_stat, p_val_t = stats.ttest_ind(trust_data, integration_data, equal_var=False)

print("\n=== Hypothesis Test Results ===")
print(f"Mann-Whitney U Test: U={u_stat}, p={p_val_mw:.4f}")
print(f"Welch's T-Test: t={t_stat:.4f}, p={p_val_t:.4f}")

# Interpretation
if p_val_mw < 0.05:
    print("Result: Statistically significant difference in control density found.")
else:
    print("Result: No statistically significant difference in control density found.")

# Visualization: Bar Chart with Error Bars (SEM)
plt.figure(figsize=(8, 6))

bundles = stats_summary.index
means = stats_summary['mean']
sems = stats_summary['sem']

colors = ['#1f77b4', '#ff7f0e'] # Blue for Integration, Orange for Trust usually, but let's just map them
bar_colors = ['skyblue' if 'Integration' in b else 'salmon' for b in bundles]

bars = plt.bar(bundles, means, yerr=sems, capsize=10, color=bar_colors, alpha=0.8)

plt.title('Average Architecture Control Density by Competency Bundle')
plt.ylabel('Avg. Number of Controls per Statement')
plt.xlabel('Competency Bundle')
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()