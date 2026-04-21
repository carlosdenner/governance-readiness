import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Categorize Lifecycle Stage
def categorize_stage(stage):
    if pd.isna(stage):
        return None
    stage_lower = str(stage).lower()
    # Operation keywords
    if any(x in stage_lower for x in ['operation', 'maintenance', 'use', 'implemented']):
        return 'Operation'
    # Development keywords
    elif any(x in stage_lower for x in ['development', 'acquisition', 'pilot', 'planning', 'research']):
        return 'Development'
    return None

eo_data['lifecycle_group'] = eo_data['16_dev_stage'].apply(categorize_stage)
analysis_df = eo_data.dropna(subset=['lifecycle_group']).copy()

# Binarize Impact Assessment
def parse_impact_assessment(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    if val_str.startswith('yes'):
        return 1
    return 0

analysis_df['has_ia'] = analysis_df['52_impact_assessment'].apply(parse_impact_assessment)

# Contingency Table
contingency_table = pd.crosstab(analysis_df['lifecycle_group'], analysis_df['has_ia'])
contingency_table.columns = ['No Impact Assessment', 'Has Impact Assessment']

print("--- Contingency Table (Counts) ---")
print(contingency_table)

# Percentages
props = pd.crosstab(analysis_df['lifecycle_group'], analysis_df['has_ia'], normalize='index') * 100
props.columns = ['No IA (%)', 'Has IA (%)']
print("\n--- Contingency Table (Percentages) ---")
print(props)

# Statistical Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio Calculation
try:
    dev_yes = contingency_table.loc['Development', 'Has Impact Assessment']
    dev_no = contingency_table.loc['Development', 'No Impact Assessment']
    op_yes = contingency_table.loc['Operation', 'Has Impact Assessment']
    op_no = contingency_table.loc['Operation', 'No Impact Assessment']
    
    # Handle zeros with pseudocount if necessary
    if dev_yes == 0 or dev_no == 0 or op_yes == 0 or op_no == 0:
        print("\n(Note: Zero counts detected, adding 0.5 correction for Odds Ratio)")
        dev_yes += 0.5
        dev_no += 0.5
        op_yes += 0.5
        op_no += 0.5

    odds_dev = dev_yes / dev_no
    odds_op = op_yes / op_no
    
    # Calculate OR comparing Operation relative to Development
    odds_ratio_op_vs_dev = odds_op / odds_dev
    
    print(f"\nOdds (Development): {odds_dev:.4f}")
    print(f"Odds (Operation): {odds_op:.4f}")
    print(f"Odds Ratio (Operation / Development): {odds_ratio_op_vs_dev:.4f}")
    
    print("\nInterpretation:")
    if p < 0.05:
        if odds_ratio_op_vs_dev > 1:
            print(f"Significant Result: Systems in Operation are {odds_ratio_op_vs_dev:.2f} times MORE likely to have an Impact Assessment than those in Development.")
        else:
            print(f"Significant Result: Systems in Operation are {1/odds_ratio_op_vs_dev:.2f} times LESS likely to have an Impact Assessment than those in Development.")
    else:
        print("No statistically significant difference found between lifecycle stages.")

except Exception as e:
    print(f"Error calculating odds ratio: {e}")