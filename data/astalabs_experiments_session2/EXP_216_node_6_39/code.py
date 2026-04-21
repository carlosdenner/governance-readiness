import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
print("Loading dataset...")
try:
    df_all = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback for different environment structures
    df_all = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
df_eo = df_all[df_all['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {df_eo.shape}")

# Define Columns
col_dev_method = '22_dev_method'
col_data_docs = '34_data_docs'
col_code_access = '38_code_access'  # Identified in debug

# 1. Define Groups (Commercial vs Internal)
# We focus on the two distinct categories identified in debug
def define_group(val):
    s = str(val).strip()
    if s == 'Developed with contracting resources.':
        return 'Commercial'
    elif s == 'Developed in-house.':
        return 'Internal'
    else:
        return None # Exclude 'Both', 'NaN', etc.

df_eo['group'] = df_eo[col_dev_method].apply(define_group)

# Filter dataset to only these two groups
df_analysis = df_eo.dropna(subset=['group']).copy()
print(f"\nAnalysis Subset Shape (Commercial + Internal only): {df_analysis.shape}")
print("Group Distribution:")
print(df_analysis['group'].value_counts())

# 2. Define Documentation Metric
def check_data_docs(val):
    if pd.isna(val):
        return False
    s = str(val).lower().strip()
    # Negative indicators
    if 'missing' in s or 'not available' in s or 'not reported' in s or s == 'no':
        return False
    # Positive indicators (completeness, partial, existing)
    if 'complete' in s or 'available' in s or 'yes' in s or 'partial' in s:
        return True
    return False

def check_code_access(val):
    if pd.isna(val):
        return False
    s = str(val).lower().strip()
    # Look for explicit Yes
    if s.startswith('yes'):
        return True
    return False

# Apply logic
df_analysis['has_data_docs'] = df_analysis[col_data_docs].apply(check_data_docs)
df_analysis['has_code_access'] = df_analysis[col_code_access].apply(check_code_access)

# Combined Metric: Has EITHER valid data docs OR code access
df_analysis['has_tech_docs'] = df_analysis['has_data_docs'] | df_analysis['has_code_access']

# 3. Statistical Analysis
# Calculate rates
group_stats = df_analysis.groupby('group')['has_tech_docs'].agg(['count', 'sum', 'mean'])
group_stats.columns = ['Total', 'With_Docs', 'Rate']
print("\nDocumentation Statistics by Group:")
print(group_stats)

# Chi-Square Test
# Contingency Table
#              Has Docs | No Docs
# Commercial      A     |    B
# Internal        C     |    D
contingency = pd.crosstab(df_analysis['group'], df_analysis['has_tech_docs'])
print("\nContingency Table:")
print(contingency)

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Results:\n  Statistic: {chi2:.4f}\n  P-value: {p:.4e}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically Significant Difference found.")
else:
    print("\nResult: No Statistically Significant Difference found.")

# 4. Visualization
plt.figure(figsize=(8, 6))

# Order: Commercial, Internal
groups = ['Commercial', 'Internal']
rates = [group_stats.loc['Commercial', 'Rate'], group_stats.loc['Internal', 'Rate']]
colors = ['#ff9999', '#66b3ff'] # Redish for commercial, Blueish for internal

bars = plt.bar(groups, rates, color=colors, edgecolor='black', alpha=0.8)

plt.ylabel('Proportion with Accessible Tech Docs')
plt.title(f"'Commercial Opacity' Effect: Technical Documentation Availability\n(Commercial vs. In-House Government AI)\np-value = {p:.4e}")
plt.ylim(0, 1.05)

# Add labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()