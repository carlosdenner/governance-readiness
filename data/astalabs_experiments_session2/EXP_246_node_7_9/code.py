import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Data Cleaning & Mapping ---

# Mapping '22_dev_method' to 'External' vs 'Internal'
def map_dev_method(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower().strip()
    
    if 'contracting resources' in val_str and 'both' not in val_str:
        return 'External/Commercial'
    elif 'in-house' in val_str and 'both' not in val_str:
        return 'Internal/Government'
    else:
        return 'Other/Mixed'

eo_df['source_group'] = eo_df['22_dev_method'].apply(map_dev_method)

# Map '52_impact_assessment' to Binary (Yes vs No)
def map_impact(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    if 'yes' in val_str:
        return 1
    return 0

eo_df['has_impact_assessment'] = eo_df['52_impact_assessment'].apply(map_impact)

# Filter for analysis: Compare External vs Internal
analysis_df = eo_df[eo_df['source_group'].isin(['External/Commercial', 'Internal/Government'])].copy()

print(f"\nRows for analysis (External vs Internal): {len(analysis_df)}")
print("Group Counts:")
print(analysis_df['source_group'].value_counts())

if len(analysis_df) > 0:
    # Contingency Table
    contingency = pd.crosstab(analysis_df['source_group'], analysis_df['has_impact_assessment'])
    # Ensure columns are ordered 0 (No), 1 (Yes) if both exist, or handle missing
    if 0 not in contingency.columns: contingency[0] = 0
    if 1 not in contingency.columns: contingency[1] = 0
    contingency = contingency[[0, 1]]
    contingency.columns = ['No Assessment', 'Has Assessment']
    
    print("\nContingency Table:")
    print(contingency)
    
    # Rates
    rates = pd.crosstab(analysis_df['source_group'], analysis_df['has_impact_assessment'], normalize='index') * 100
    print("\nCompliance Rates (%):")
    print(rates)
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\n--- Statistical Test Results ---")
    print(f"Chi2: {chi2:.4f}, p-value: {p:.5f}")
    
    # Interpretation
    ext_rate = rates.loc['External/Commercial', 1] if 1 in rates.columns else 0
    int_rate = rates.loc['Internal/Government', 1] if 1 in rates.columns else 0
    
    print(f"\nCompare: External ({ext_rate:.2f}%) vs Internal ({int_rate:.2f}%)")
    
    if p < 0.05:
        print("Result: Significant Difference.")
        if ext_rate < int_rate:
            print("Direction: External < Internal (Supports Hypothesis: Commercial Gap)")
        else:
            print("Direction: External > Internal (Contradicts Hypothesis)")
    else:
        print("Result: No Significant Difference (Null Hypothesis retained).")
else:
    print("Insufficient data for comparison.")
