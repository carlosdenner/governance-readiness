import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

# Define dataset path
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print("Error: Dataset not found.")
    sys.exit(1)

# Filter for EO 13960 Scored dataset
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {subset.shape}")

# Identify columns dynamically
use_case_col = next((c for c in subset.columns if 'use_case_name' in c), None)
purpose_col = next((c for c in subset.columns if 'purpose_benefits' in c), None)
mitigation_col = next((c for c in subset.columns if 'disparity_mitigation' in c), None)

print(f"Columns identified: Use Case='{use_case_col}', Purpose='{purpose_col}', Mitigation='{mitigation_col}'")

if not (use_case_col and purpose_col and mitigation_col):
    print("Critical columns missing. Aborting.")
    sys.exit(1)

# --- 1. Define Biometric Systems ---
# Keywords for biometrics
bio_keywords = ['face', 'facial', 'biometric', 'recognition', 'surveillance', 'gait', 'iris']

# Combine text for searching
subset['text_search'] = subset[use_case_col].fillna('').astype(str).str.lower() + " " + subset[purpose_col].fillna('').astype(str).str.lower()

# Apply boolean mask
subset['Is_Biometric'] = subset['text_search'].apply(lambda x: any(k in x for k in bio_keywords))

print(f"Biometric systems found: {subset['Is_Biometric'].sum()} out of {len(subset)}")

# --- 2. Define Mitigation Presence ---
# Positive keywords indicating some form of check/control exists
mitigation_keywords = ['test', 'eval', 'monitor', 'assess', 'audit', 'mitigat', 'review', 'human', 'valid', 'bias', 'fair', 'check', 'control', 'feedback']

# Function to check mitigation text
def check_mitigation(text):
    if pd.isna(text):
        return False
    text = str(text).lower()
    # Check for positive keywords
    has_positive = any(k in text for k in mitigation_keywords)
    return has_positive

subset['Has_Mitigation'] = subset[mitigation_col].apply(check_mitigation)

print(f"Systems with Mitigation found: {subset['Has_Mitigation'].sum()} out of {len(subset)}")

# --- 3. Statistical Analysis ---
contingency = pd.crosstab(subset['Is_Biometric'], subset['Has_Mitigation'])
print("\nContingency Table (Rows=Is_Biometric, Cols=Has_Mitigation):")
print(contingency)

# Calculate rates safely
try:
    # Biometric stats
    bio_total = contingency.loc[True].sum() if True in contingency.index else 0
    bio_mitigated = contingency.loc[True, True] if (True in contingency.index and True in contingency.columns) else 0
    bio_rate = bio_mitigated / bio_total if bio_total > 0 else 0.0

    # Non-Biometric stats
    non_bio_total = contingency.loc[False].sum() if False in contingency.index else 0
    non_bio_mitigated = contingency.loc[False, True] if (False in contingency.index and True in contingency.columns) else 0
    non_bio_rate = non_bio_mitigated / non_bio_total if non_bio_total > 0 else 0.0

    print(f"\nBiometric Mitigation Rate: {bio_rate:.2%}")
    print(f"Non-Biometric Mitigation Rate: {non_bio_rate:.2%}")

except Exception as e:
    print(f"Error calculating rates: {e}")

# Perform Chi-Square Test
if contingency.size >= 4:
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Result:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")
else:
    print("\nContingency table is too small for Chi-Square test (needs 2x2).")
