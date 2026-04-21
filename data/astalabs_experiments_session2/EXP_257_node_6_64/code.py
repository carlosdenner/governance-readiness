import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import sys
import os

# --- Step 1: Load the dataset ---
file_name = 'astalabs_discovery_all_data.csv'
file_path = file_name
if not os.path.exists(file_path):
    file_path = f'../{file_name}'
    if not os.path.exists(file_path):
        print(f"Error: File {file_name} not found.")
        sys.exit(1)

print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path, low_memory=False)

# Filter for the relevant source table
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded {len(df_eo)} rows from EO 13960 subset.")

# --- Step 2: Data Cleaning ---

# Independent Variable: Commercial vs Custom (Proxy: 37_custom_code)
# 'No' -> No custom code -> Commercial/COTS
# 'Yes' -> Custom code -> Custom/In-house
def clean_commercial_proxy(val):
    s = str(val).strip().lower()
    if s == 'no':
        return 'Commercial'
    elif s == 'yes':
        return 'Custom/In-house'
    return None

# Dependent Variable 1: Code Access (38_code_access)
# 'Yes...' -> Yes, 'No...' -> No
def clean_code_access(val):
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 'Yes'
    # Treat No, nan, blank, or other descriptions as 'No'
    return 'No'

# Dependent Variable 2: Impact Assessment (52_impact_assessment)
# 'Yes' -> Yes, 'No' or 'Planned' -> No
def clean_impact_assessment(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 'Yes'
    # 'planned or in-progress' counts as No for "completed" assessment
    return 'No'

# Apply cleaning
df_eo['Commercial_Status'] = df_eo['37_custom_code'].apply(clean_commercial_proxy)
df_eo['Has_Code_Access'] = df_eo['38_code_access'].apply(clean_code_access)
df_eo['Has_Impact_Assessment'] = df_eo['52_impact_assessment'].apply(clean_impact_assessment)

# Filter to valid groups
df_clean = df_eo.dropna(subset=['Commercial_Status'])

print(f"Analyzable use cases after cleaning: {len(df_clean)}")
print("Group Distribution:")
print(df_clean['Commercial_Status'].value_counts())

# --- Step 3: Analysis - Commercial vs Code Access ---
print("\n=======================================================")
print("TEST 1: Commercial Status vs. Code Access")
print("=======================================================")

ct_code = pd.crosstab(df_clean['Commercial_Status'], df_clean['Has_Code_Access'])
print("Contingency Table (Code Access):")
print(ct_code)

chi2_code, p_code, dof_code, ex_code = chi2_contingency(ct_code)
print(f"\nChi-Square Statistic: {chi2_code:.4f}")
print(f"P-Value: {p_code:.4e}")

# Calculate Odds Ratio (Odds of Commercial having Access / Odds of Custom having Access)
try:
    # Add small smoothing to avoid div by zero if necessary
    smoothing = 0.5 if (ct_code == 0).any().any() else 0
    
    n_comm_yes = ct_code.loc['Commercial', 'Yes'] + smoothing
    n_comm_no = ct_code.loc['Commercial', 'No'] + smoothing
    n_cust_yes = ct_code.loc['Custom/In-house', 'Yes'] + smoothing
    n_cust_no = ct_code.loc['Custom/In-house', 'No'] + smoothing
    
    odds_comm = n_comm_yes / n_comm_no
    odds_cust = n_cust_yes / n_cust_no
    
    or_code = odds_comm / odds_cust
    print(f"Odds Ratio (Commercial vs Custom): {or_code:.4f}")
    if or_code < 1:
        print(f"Interpretation: Commercial systems are {1/or_code:.2f}x LESS likely to have Code Access.")
except Exception as e:
    print(f"Could not calculate Odds Ratio: {e}")

# --- Step 4: Analysis - Commercial vs Impact Assessment ---
print("\n=======================================================")
print("TEST 2: Commercial Status vs. Impact Assessment")
print("=======================================================")

ct_impact = pd.crosstab(df_clean['Commercial_Status'], df_clean['Has_Impact_Assessment'])
print("Contingency Table (Impact Assessment):")
print(ct_impact)

chi2_impact, p_impact, dof_impact, ex_impact = chi2_contingency(ct_impact)
print(f"\nChi-Square Statistic: {chi2_impact:.4f}")
print(f"P-Value: {p_impact:.4e}")

# Calculate Odds Ratio
try:
    smoothing = 0.5 if (ct_impact == 0).any().any() else 0
    
    n_comm_yes = ct_impact.loc['Commercial', 'Yes'] + smoothing
    n_comm_no = ct_impact.loc['Commercial', 'No'] + smoothing
    n_cust_yes = ct_impact.loc['Custom/In-house', 'Yes'] + smoothing
    n_cust_no = ct_impact.loc['Custom/In-house', 'No'] + smoothing
    
    odds_comm = n_comm_yes / n_comm_no
    odds_cust = n_cust_yes / n_cust_no
    
    or_impact = odds_comm / odds_cust
    print(f"Odds Ratio (Commercial vs Custom): {or_impact:.4f}")
    if or_impact < 1:
        print(f"Interpretation: Commercial systems are {1/or_impact:.2f}x LESS likely to have Impact Assessments.")
except Exception as e:
    print(f"Could not calculate Odds Ratio: {e}")
