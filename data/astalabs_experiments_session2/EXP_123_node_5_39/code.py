import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback for local testing if needed, though strictly we use the provided filename
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Define columns
col_notice = '59_ai_notice'
col_appeal = '65_appeal_process'

print(f"Processing {len(df_eo)} records from EO 13960 subset.")

# 1. Clean '59_ai_notice'
# Logic: 'Online', 'In-person', 'Yes' -> Yes. 'N/A', 'No' -> No.
def clean_notice(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    # Positive indicators
    if any(x in s for x in ['online', 'in-person', 'yes', 'public']):
        return 'Yes'
    # Negative indicators
    if any(x in s for x in ['n/a', 'no', 'none', 'not applicable']):
        return 'No'
    return np.nan

# 2. Clean '65_appeal_process'
# Logic: Starts with 'Yes' -> Yes. Starts with 'No', 'N/A' -> No.
def clean_appeal(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    if s.startswith('yes'):
        return 'Yes'
    if s.startswith('no') or s.startswith('n/a') or 'not applicable' in s:
        return 'No'
    return np.nan

# Apply cleaning
df_eo['Notice_Clean'] = df_eo[col_notice].apply(clean_notice)
df_eo['Appeal_Clean'] = df_eo[col_appeal].apply(clean_appeal)

# Drop rows where either value is NaN to ensure valid comparison
df_analysis = df_eo.dropna(subset=['Notice_Clean', 'Appeal_Clean'])

print(f"Valid records after cleaning: {len(df_analysis)}")

if len(df_analysis) > 0:
    # Create Contingency Table
    contingency = pd.crosstab(df_analysis['Notice_Clean'], df_analysis['Appeal_Clean'])
    
    # Reindex to ensure consistent order (Yes/No)
    desired_index = [x for x in ['Yes', 'No'] if x in contingency.index]
    desired_cols = [x for x in ['Yes', 'No'] if x in contingency.columns]
    contingency = contingency.loc[desired_index, desired_cols]
    
    print("\nContingency Table (Rows: Notice, Cols: Appeal):")
    print(contingency)
    
    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Conditional Probabilities
    # P(Appeal=Yes | Notice=Yes)
    if 'Yes' in contingency.index and 'Yes' in contingency.columns:
        n_notice_yes = contingency.loc['Yes'].sum()
        n_both_yes = contingency.loc['Yes', 'Yes']
        if n_notice_yes > 0:
            print(f"Rate of Appeal Process when Notice is provided: {n_both_yes/n_notice_yes:.1%} ({n_both_yes}/{n_notice_yes})")
    
    # P(Appeal=Yes | Notice=No)
    if 'No' in contingency.index and 'Yes' in contingency.columns:
        n_notice_no = contingency.loc['No'].sum()
        n_appeal_yes_notice_no = contingency.loc['No', 'Yes']
        if n_notice_no > 0:
            print(f"Rate of Appeal Process when Notice is NOT provided: {n_appeal_yes_notice_no/n_notice_no:.1%} ({n_appeal_yes_notice_no}/{n_notice_no})")
            
else:
    print("No valid records found after cleaning. Check value patterns.")
    print("Sample Notice values:", df_eo[col_notice].unique()[:5])
    print("Sample Appeal values:", df_eo[col_appeal].unique()[:5])