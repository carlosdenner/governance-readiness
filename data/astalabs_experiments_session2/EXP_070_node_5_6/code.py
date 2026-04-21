import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    # Reading with low_memory=False to avoid mixed type warnings
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Robust Processing Functions ---

def categorize_stage(val):
    # Robustly convert to string and lower case to handle NaNs (floats)
    s = str(val).lower()
    if 'operation' in s or 'maintenance' in s:
        return 'Operation'
    if 'retired' in s:
        # Retired systems are post-operational, grouping with Operation for 'Legacy' context
        return 'Operation'
    return 'Development'

def parse_docs(val):
    # Robustly convert to string and normalize
    s = str(val).lower().strip()
    # Define values that map to False/0 (missing documentation)
    if s in ['nan', 'no', 'none', 'n/a', '0', 'false', '']:
        return 0
    return 1

# Apply categorization directly to source columns
eo_data['stage_group'] = eo_data['16_dev_stage'].apply(categorize_stage)
eo_data['has_docs'] = eo_data['34_data_docs'].apply(parse_docs)

# --- Analysis ---

# Calculate rates
summary = eo_data.groupby('stage_group')['has_docs'].agg(['count', 'sum', 'mean'])
summary.rename(columns={'count': 'Total', 'sum': 'With Docs', 'mean': 'Rate'}, inplace=True)

print("--- Summary Statistics: Data Documentation by Stage ---")
print(summary)
print("\n")

# Contingency Table
contingency_table = pd.crosstab(eo_data['stage_group'], eo_data['has_docs'])
print("--- Contingency Table ---")
print(contingency_table)
print("\n")

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"--- Chi-Square Test Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant Difference found.")
    if summary.loc['Operation', 'Rate'] > summary.loc['Development', 'Rate']:
        print("Hypothesis Supported: Operation stage has significantly higher documentation rates.")
    else:
        print("Hypothesis Contradicted: Operation stage does not have higher documentation rates.")
else:
    print("Result: No Statistically Significant Difference found.")

# --- Visualization ---
plt.figure(figsize=(8, 6))

# Define colors
bar_colors = ['#1f77b4' if idx == 'Development' else '#ff7f0e' for idx in summary.index]

bars = plt.bar(summary.index, summary['Rate'], color=bar_colors, edgecolor='black', alpha=0.8)

plt.title('Data Documentation Availability by Development Stage')
plt.ylabel('Proportion of Projects with Documentation')
plt.xlabel('Project Stage')
plt.ylim(0, 1.15)

# Add labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()