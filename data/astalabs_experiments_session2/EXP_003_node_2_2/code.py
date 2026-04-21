import pandas as pd
import scipy.stats as stats
import numpy as np
import os

# Define file path
file_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = '../astalabs_discovery_all_data.csv'

# Load dataset
print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 Scored
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset shape: {subset.shape}")

# --- Step 1: Analyze and Create 'is_commercial' ---
print("\n--- Value Counts for '10_commercial_ai' (Top 5) ---")
print(subset['10_commercial_ai'].value_counts(dropna=False).head(5))

# Logic: 'None of the above.' = Custom (0), Any other text = Commercial (1)
def map_commercial(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str == "None of the above.":
        return 0 # Custom
    else:
        return 1 # Commercial

subset['is_commercial'] = subset['10_commercial_ai'].apply(map_commercial)
subset = subset.dropna(subset=['is_commercial'])

print("\n--- Commercial vs Custom Counts ---")
print(subset['is_commercial'].value_counts())

# --- Step 2: Analyze and Create Compliance Flags ---
# Inspect values to determine binarization logic
print("\n--- Value Counts for '52_impact_assessment' ---")
print(subset['52_impact_assessment'].value_counts(dropna=False).head())
print("\n--- Value Counts for '55_independent_eval' ---")
print(subset['55_independent_eval'].value_counts(dropna=False).head())

def binarize_compliance(val):
    if pd.isna(val):
        return 0 # Treat missing as No
    val_str = str(val).lower().strip()
    if val_str.startswith('yes'):
        return 1
    return 0

subset['has_impact_assess'] = subset['52_impact_assessment'].apply(binarize_compliance)
subset['has_indep_eval'] = subset['55_independent_eval'].apply(binarize_compliance)

# --- Step 3: Statistical Analysis ---
def run_stats(df, target_col, label):
    print(f"\n=== Analysis for: {label} ===")
    
    # Contingency Table
    # Rows: 0=Custom, 1=Commercial
    # Cols: 0=No, 1=Yes
    ct = pd.crosstab(df['is_commercial'], df[target_col])
    
    # Ensure 2x2
    ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    print("Contingency Table (Rows: 0=Custom, 1=Comm; Cols: 0=No, 1=Yes):")
    print(ct)
    
    # Check if we have enough data
    if (ct.values == 0).any():
        print("Warning: Zero values in contingency table. Adding correction (0.5) for Odds Ratio.")
        ct_adj = ct + 0.5
    else:
        ct_adj = ct
        
    # Chi-Square
    # Use the original table for Chi2, unless margins are 0
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
    except ValueError as e:
        print(f"Chi-Square failed: {e}")
        chi2, p = 0, 1

    # Odds Ratio Calculation
    # OR = (Odds Commercial) / (Odds Custom)
    # Odds = Yes / No
    # OR = (Comm_Yes/Comm_No) / (Cust_Yes/Cust_No)
    
    comm_yes = ct_adj.loc[1, 1]
    comm_no = ct_adj.loc[1, 0]
    cust_yes = ct_adj.loc[0, 1]
    cust_no = ct_adj.loc[0, 0]
    
    odds_comm = comm_yes / comm_no
    odds_cust = cust_yes / cust_no
    or_val = odds_comm / odds_cust
    
    # Output
    print(f"Compliance Rate (Custom):     {cust_yes/(cust_yes+cust_no):.2%}")
    print(f"Compliance Rate (Commercial): {comm_yes/(comm_yes+comm_no):.2%}")
    print(f"Chi-Square Statistic:         {chi2:.4f}")
    print(f"P-Value:                      {p:.5f}")
    print(f"Odds Ratio (Comm vs Cust):    {or_val:.4f}")
    
    if p < 0.05:
        print("Result: Significant difference detected.")
    else:
        print("Result: No significant difference.")

run_stats(subset, 'has_impact_assess', 'Impact Assessment (Control 52)')
run_stats(subset, 'has_indep_eval', 'Independent Evaluation (Control 55)')
