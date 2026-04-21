import pandas as pd
import scipy.stats as stats
import numpy as np
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for relevant source table
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset size: {len(subset)}")

# Define columns
stage_col = '16_dev_stage'
mitigation_col = '62_disparity_mitigation'

# -- Data Cleaning & Categorization --

# 1. Categorize Development Stage
def categorize_stage(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower()
    # Operational keywords
    if any(x in val for x in ['operation', 'maintenance', 'use', 'production', 'deployed', 'mission']):
        return 'Operational'
    # Pre-Operational keywords
    elif any(x in val for x in ['development', 'acquisition', 'plan', 'design', 'pilot', 'test', 'initiat', 'implement']):
        return 'Pre-Operational'
    return 'Other'

subset['stage_category'] = subset[stage_col].apply(categorize_stage)

# 2. Categorize Disparity Mitigation
# Logic: Treat explicit "N/A", "No", "None", or Null as 0. Treat substantive descriptions as 1.
def categorize_mitigation(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    
    # Check for negatives
    if val_str.startswith(('n/a', 'no ', 'none', 'not applicable', 'not safety')):
        return 0
    if val_str == 'no':
        return 0
    
    # Check for positives (heuristics based on text analysis)
    # If it's not N/A and has some length, it's likely a description of a control or process
    if len(val_str) > 3:
        return 1
        
    return 0

subset['mitigation_binary'] = subset[mitigation_col].apply(categorize_mitigation)

# Filter analysis set
analysis_df = subset[subset['stage_category'].isin(['Operational', 'Pre-Operational'])].copy()
print(f"Analysis set size: {len(analysis_df)}")
print("Stage distribution:\n", analysis_df['stage_category'].value_counts())
print("Mitigation distribution:\n", analysis_df['mitigation_binary'].value_counts())

# -- Analysis --

# Contingency Table
contingency = pd.crosstab(analysis_df['stage_category'], analysis_df['mitigation_binary'])
print("\nContingency Table (Count):")
print(contingency)

# Check if we have both 0 and 1 columns
if 1 not in contingency.columns:
    print("\nError: No positive mitigation cases found in the filtered dataset. Cannot perform Chi-square test.")
    # Debug print to see what text triggered this if any
    print("Sample of mitigation text that mapped to 0:")
    print(subset[subset['mitigation_binary']==0][mitigation_col].head(5))
    sys.exit(0)

if 0 not in contingency.columns:
    # Unlikely given the data, but possible
    contingency[0] = 0

# Reorder columns for consistency [0, 1]
contingency = contingency[[0, 1]]
contingency.columns = ['No Mitigation', 'Has Mitigation']

# Calculate percentages
props = contingency.div(contingency.sum(axis=1), axis=0) * 100
print("\nContingency Table (Percentages):")
print(props)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio
try:
    # OR = (Op_Yes / Op_No) / (Pre_Yes / Pre_No)
    op_yes = contingency.loc['Operational', 'Has Mitigation']
    op_no = contingency.loc['Operational', 'No Mitigation']
    pre_yes = contingency.loc['Pre-Operational', 'Has Mitigation']
    pre_no = contingency.loc['Pre-Operational', 'No Mitigation']
    
    # Add small epsilon if 0 to avoid division by zero error
    if op_no == 0: op_no = 0.5
    if pre_no == 0: pre_no = 0.5
    if pre_yes == 0: pre_yes = 0.5
    
    odds_op = op_yes / op_no
    odds_pre = pre_yes / pre_no
    odds_ratio = odds_op / odds_pre
    
    print(f"\nOdds (Operational): {odds_op:.4f}")
    print(f"Odds (Pre-Operational): {odds_pre:.4f}")
    print(f"Odds Ratio (Operational / Pre-Operational): {odds_ratio:.4f}")
except Exception as e:
    print(f"\nError calculating OR: {e}")

# Interpret
if p < 0.05:
    print("\nConclusion: Statistically significant difference.")
    if props.loc['Operational', 'Has Mitigation'] < props.loc['Pre-Operational', 'Has Mitigation']:
        print("Direction: Operational systems are LESS likely to report mitigation (Supporting Hypothesis).")
    else:
        print("Direction: Operational systems are MORE likely to report mitigation (Refuting Hypothesis).")
else:
    print("\nConclusion: No statistically significant difference.")