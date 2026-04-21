import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define the filename
filename = 'astalabs_discovery_all_data.csv'

# Check if file exists in current directory, if not try ../, if not fail gracefully
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    # Fallback to absolute path check or just assume current dir if all else fails to let pandas handle error
    filepath = filename

print(f"Loading dataset from {filepath}...")
try:
    df = pd.read_csv(filepath, low_memory=False)
except FileNotFoundError:
    # If the logic above failed, try one last desperate attempt based on the prompt note
    filepath = '../astalabs_discovery_all_data.csv'
    df = pd.read_csv(filepath, low_memory=False)

# Filter for EO13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 subset shape: {eo_df.shape}")

# Columns of interest
stage_col = '16_dev_stage'
monitor_col = '56_monitor_postdeploy'
# Additional columns for composite governance score
gov_cols = ['56_monitor_postdeploy', '55_independent_eval', '30_saop_review', '52_impact_assessment']

# Function to clean boolean-like columns
def parse_bool(x):
    if pd.isna(x):
        return 0
    s = str(x).lower().strip()
    if s in ['yes', 'true', '1', 'y']:
        return 1
    return 0

# Apply cleaning
for col in gov_cols:
    if col in eo_df.columns:
        eo_df[col + '_score'] = eo_df[col].apply(parse_bool)
    else:
        print(f"Warning: Column {col} not found. Filling with 0.")
        eo_df[col + '_score'] = 0

# Calculate scores
eo_df['Monitoring_Score'] = eo_df[monitor_col + '_score']
eo_df['Governance_Score'] = eo_df[[c + '_score' for c in gov_cols]].mean(axis=1)

# Inspect Stages
print("\nUnique Lifecycle Stages found:")
print(eo_df[stage_col].value_counts())

# Define groups
# We treat 'Operational' and 'Use' as Operational.
# We treat 'Development' and 'Research' as Development.
def categorize_stage(val):
    s = str(val).lower()
    if 'oper' in s or 'use' in s or 'maintenance' in s:
        return 'Operational'
    if 'dev' in s or 'research' in s or 'pilot' in s:
        return 'Development'
    return 'Other'

eo_df['Stage_Category'] = eo_df[stage_col].apply(categorize_stage)

# Filter for only Operational and Development
analysis_df = eo_df[eo_df['Stage_Category'].isin(['Operational', 'Development'])].copy()

# Calculate stats
stats_df = analysis_df.groupby('Stage_Category')[['Monitoring_Score', 'Governance_Score']].agg(['mean', 'std', 'count'])
print("\nStatistics by Stage Category:")
print(stats_df)

# T-Test
op_scores = analysis_df[analysis_df['Stage_Category'] == 'Operational']['Monitoring_Score']
dev_scores = analysis_df[analysis_df['Stage_Category'] == 'Development']['Monitoring_Score']

t_stat, p_val = stats.ttest_ind(op_scores, dev_scores, equal_var=False)
print(f"\nT-test (Monitoring Score): Operational vs Development")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4e}")

if p_val < 0.05:
    print("Result: Significant difference detected.")
    if t_stat < 0:
        print("Direction: Operational systems have LOWER monitoring compliance than Development systems.")
    else:
        print("Direction: Operational systems have HIGHER monitoring compliance than Development systems.")
else:
    print("Result: No significant difference detected.")

# Visualization
plt.figure(figsize=(10, 6))
means = stats_df['Monitoring_Score']['mean']
errors = stats_df['Monitoring_Score']['std'] / np.sqrt(stats_df['Monitoring_Score']['count'])

# Ensure order is Development then Operational for logical flow
plot_order = ['Development', 'Operational']
means = means.reindex(plot_order)
errors = errors.reindex(plot_order)

bars = plt.bar(plot_order, means, yerr=errors, capsize=10, color=['#ff9999', '#66b3ff'], alpha=0.9)
plt.title('Compliance with "Continuous Monitoring" by Lifecycle Stage')
plt.ylabel('Proportion of Systems Compliant')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Secondary Plot: Composite Governance Score
plt.figure(figsize=(10, 6))
comp_means = stats_df['Governance_Score']['mean'].reindex(plot_order)
comp_errors = (stats_df['Governance_Score']['std'] / np.sqrt(stats_df['Governance_Score']['count'])).reindex(plot_order)

bars2 = plt.bar(plot_order, comp_means, yerr=comp_errors, capsize=10, color=['#99ff99', '#ffcc99'], alpha=0.9)
plt.title('Composite Governance Score (Eval + Monitor + Review) by Stage')
plt.ylabel('Average Score (0-1)')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()