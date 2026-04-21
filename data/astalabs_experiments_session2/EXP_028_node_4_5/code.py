import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
subset = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Data Loading ---")
print(f"EO 13960 Records: {len(subset)}")

# Inspect Independent Evaluation Column
print("\n--- Unique Values in '55_independent_eval' ---")
unique_evals = subset['55_independent_eval'].unique()
print(unique_evals)

# Define categorization function for Lifecycle Stage
def categorize_stage(val):
    s = str(val).lower()
    if 'retired' in s:
        return 'Retired'
    # Operation keywords
    op_keywords = ['operation', 'maintenance', 'production', 'mission', 'implementation', 'deployed', 'use']
    if any(k in s for k in op_keywords):
        return 'Operation'
    # Pre-Operation keywords
    pre_op_keywords = ['acquisition', 'development', 'initiated', 'planned', 'pilot', 'research', 'design', 'testing']
    if any(k in s for k in pre_op_keywords):
        return 'Pre-Operation'
    return 'Other'

subset['stage_group'] = subset['16_dev_stage'].apply(categorize_stage)

# Filter for relevant groups
analysis_df = subset[subset['stage_group'].isin(['Operation', 'Pre-Operation'])].copy()

# Binarize Independent Evaluation
# Robust check for affirmative values
def is_affirmative(val):
    s = str(val).lower().strip()
    # Check for Yes, True, or 1
    if s in ['yes', 'true', '1', '1.0']:
        return 1
    # Check for "completed" or similar if inspection reveals it
    if 'yes' in s:
        return 1
    return 0

analysis_df['has_eval'] = analysis_df['55_independent_eval'].apply(is_affirmative)

# Calculate Rates
group_stats = analysis_df.groupby('stage_group')['has_eval'].agg(['count', 'sum', 'mean'])
group_stats['mean_pct'] = group_stats['mean'] * 100
print("\n--- Independent Evaluation Statistics ---")
print(group_stats)

# Extract counts for Chi-Square
# Table format: [[Op_Yes, Op_No], [Pre_Yes, Pre_No]]
op_yes = group_stats.loc['Operation', 'sum']
op_total = group_stats.loc['Operation', 'count']
op_no = op_total - op_yes

pre_yes = group_stats.loc['Pre-Operation', 'sum']
pre_total = group_stats.loc['Pre-Operation', 'count']
pre_no = pre_total - pre_yes

contingency_table = [[op_yes, op_no], [pre_yes, pre_no]]

print("\n--- Contingency Table (Yes, No) ---")
print(f"Operation:     {op_yes}, {op_no}")
print(f"Pre-Operation: {pre_yes}, {pre_no}")

# Check for validity of Chi-Square
total_yes = op_yes + pre_yes
if total_yes == 0:
    print("\nRESULT: No independent evaluations found in the dataset (0 positive cases).")
    print("Cannot perform Chi-Square test.")
else:
    # Perform Chi-Square Test
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")

    if p < 0.05:
        print("Result: Statistically significant difference between stages.")
    else:
        print("Result: No statistically significant difference.")

    # Visualization
    plt.figure(figsize=(8, 6))
    bars = plt.bar(group_stats.index, group_stats['mean_pct'], color=['#2ca02c', '#1f77b4'])
    plt.title('Independent Evaluation Rate by Lifecycle Stage')
    plt.ylabel('Percentage with Independent Eval (%)')
    plt.xlabel('Lifecycle Stage')
    # Dynamic ylim
    ymax = group_stats['mean_pct'].max()
    if ymax == 0:
        ymax = 10
    plt.ylim(0, ymax * 1.2)

    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (ymax*0.02),
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()