import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Function to load dataset
def load_dataset(filename):
    paths = [filename, os.path.join('..', filename)]
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find {filename} in {paths}")

# 1. Load Data
df = load_dataset('step3_tactic_frequency.csv')

# 2. Data Processing
# Group by sub_competency_id and bundle, then count unique tactic_ids
tactic_counts = df.groupby(['sub_competency_id', 'bundle'])['tactic_id'].nunique().reset_index()
tactic_counts.rename(columns={'tactic_id': 'unique_tactic_count'}, inplace=True)

print("Summary of Tactic Counts per Sub-Competency:")
print(tactic_counts)

# 3. Statistical Test
trust_group = tactic_counts[tactic_counts['bundle'] == 'Trust Readiness']['unique_tactic_count']
integration_group = tactic_counts[tactic_counts['bundle'] == 'Integration Readiness']['unique_tactic_count']

print(f"\nTrust Readiness (n={len(trust_group)}): Mean = {trust_group.mean():.2f}, Std = {trust_group.std():.2f}")
print(f"Integration Readiness (n={len(integration_group)}): Mean = {integration_group.mean():.2f}, Std = {integration_group.std():.2f}")

# T-test (independent samples)
# Using Welch's t-test (equal_var=False) due to small sample sizes and potential variance differences
t_stat, p_val = stats.ttest_ind(trust_group, integration_group, equal_var=False)
print(f"\nT-test results: Statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

# 4. Visualization
plt.figure(figsize=(10, 6))
# Fix for seaborn warning: assign x to hue and set legend=False
sns.violinplot(x='bundle', y='unique_tactic_count', hue='bundle', data=tactic_counts, inner='stick', palette='muted', legend=False)
plt.title('Distribution of Unique Adversarial Tactics per Competency Bundle')
plt.ylabel('Count of Unique Tactics')
plt.xlabel('Competency Bundle')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()