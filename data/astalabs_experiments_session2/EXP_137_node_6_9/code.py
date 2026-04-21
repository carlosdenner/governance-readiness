import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Dynamic file path resolution
filename = 'astalabs_discovery_all_data.csv'
filepath = filename
if not os.path.exists(filepath):
    if os.path.exists('../' + filename):
        filepath = '../' + filename
    else:
        print(f"Warning: {filename} not found in current or parent directory. Attempting current directory default.")

print(f"Loading dataset from {filepath}...")
try:
    df = pd.read_csv(filepath, low_memory=False)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(1)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 records loaded: {len(eo_df)}")

# Target Columns
col_dev = '22_dev_method'
col_notice = '59_ai_notice'

# Normalize and clean '22_dev_method'
# We want to distinguish 'In-house' vs 'Contracted'
def clean_dev_method(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    # Check for both/hybrid first if necessary, but hypothesis compares the two distinct groups.
    # Common logic: if it mentions contract, it involves contractors.
    # If it mentions in-house ONLY, it's in-house.
    if 'contract' in s:
        # specific check for hybrid could go here, but let's classify as 'Contracted' for now or 'Hybrid'
        if 'in-house' in s or 'government' in s:
             return 'Hybrid/Both'
        return 'Contracted'
    if 'in-house' in s or 'government' in s:
        return 'In-house'
    return 'Other'

eo_df['dev_category'] = eo_df[col_dev].apply(clean_dev_method)

# Normalize and clean '59_ai_notice'
def clean_notice(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    if 'yes' in s:
        return 'Yes'
    if 'no' in s:
        return 'No'
    return np.nan

eo_df['notice_flag'] = eo_df[col_notice].apply(clean_notice)

# Filter data for the hypothesis test (In-house vs Contracted)
# Excluding Hybrid/Other to test the specific friction between purely in-house vs contracted.
analysis_df = eo_df[eo_df['dev_category'].isin(['In-house', 'Contracted'])].copy()
analysis_df = analysis_df.dropna(subset=['notice_flag'])

print(f"\nRecords for analysis (In-house vs Contracted, valid notice): {len(analysis_df)}")
print("Distribution by Development Method:")
print(analysis_df['dev_category'].value_counts())

# Contingency Table
contingency = pd.crosstab(analysis_df['dev_category'], analysis_df['notice_flag'])
print("\nContingency Table (Dev Method vs AI Notice):")
print(contingency)

# Calculate percentages
props = pd.crosstab(analysis_df['dev_category'], analysis_df['notice_flag'], normalize='index') * 100
print("\nProportions (%):")
print(props)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Test Results:")
print(f"Chi2: {chi2:.4f}, p-value: {p:.4e}")

# Visualization
if not props.empty:
    # Sorting to ensure consistent order if needed, or just relying on index
    ax = props.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#d62728', '#2ca02c'], alpha=0.8)
    plt.title('AI Notice Compliance: In-house vs Contracted')
    plt.xlabel('Development Method')
    plt.ylabel('Percentage')
    plt.legend(title='Notice Provided', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=0)
    
    # Annotate
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')
    
    plt.tight_layout()
    plt.show()
