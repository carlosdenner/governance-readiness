import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define file path based on instructions (one level above)
file_path = '../step3_incident_coding.csv'

# Fallback to current directory if not found in parent (robustness)
if not os.path.exists(file_path) and os.path.exists('step3_incident_coding.csv'):
    file_path = 'step3_incident_coding.csv'

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    # Attempting to list directories to aid debugging if this fails
    print(f"Contents of ..: {os.listdir('..')}")
    raise

# Preprocessing
# Ensure technique_count is numeric
df['technique_count'] = pd.to_numeric(df['technique_count'], errors='coerce')

# Create binary category
# 'security' vs everything else
df['Harm_Category'] = df['harm_type'].apply(lambda x: 'Security' if str(x).lower() == 'security' else 'Other')

# Descriptive Statistics
group_stats = df.groupby('Harm_Category')['technique_count'].agg(['count', 'mean', 'std', 'sem'])
print("\n=== Descriptive Statistics (Technique Count) ===")
print(group_stats)

# Prepare data for testing
sec_data = df[df['Harm_Category'] == 'Security']['technique_count'].dropna()
other_data = df[df['Harm_Category'] == 'Other']['technique_count'].dropna()

# Statistical Testing
# 1. Mann-Whitney U Test (Non-parametric, robust to outliers/non-normality)
u_stat, p_val_mw = stats.mannwhitneyu(sec_data, other_data, alternative='two-sided')

# 2. Welch's T-test (Parametric, assumes normality but robust to unequal variances)
t_stat, p_val_t = stats.ttest_ind(sec_data, other_data, equal_var=False)

print("\n=== Statistical Test Results ===")
print(f"Mann-Whitney U Test: Statistic={u_stat}, p-value={p_val_mw:.5f}")
print(f"Welch's T-Test:      Statistic={t_stat:.4f}, p-value={p_val_t:.5f}")

# Interpretation helper
if p_val_mw < 0.05:
    print("Result: Statistically significant difference detected (p < 0.05).")
else:
    print("Result: No statistically significant difference detected (p >= 0.05).")

# Visualization
plt.figure(figsize=(8, 6))

# Reorder for consistent plotting
plot_order = ['Security', 'Other']
plot_means = [group_stats.loc[cat, 'mean'] for cat in plot_order]
plot_sems = [group_stats.loc[cat, 'sem'] for cat in plot_order]
plot_colors = ['salmon', 'lightgray']

bars = plt.bar(plot_order, plot_means, yerr=plot_sems, capsize=10, color=plot_colors, edgecolor='black', alpha=0.8)

plt.title('Mean ATLAS Technique Count by Harm Category')
plt.ylabel('Mean Technique Count (+/- SEM)')
plt.xlabel('Harm Category')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{height:.2f}',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()