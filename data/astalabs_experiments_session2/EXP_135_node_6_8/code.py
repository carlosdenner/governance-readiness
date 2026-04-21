import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# Define file path
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, f'../{filename}']
filepath = next((p for p in possible_paths if os.path.exists(p)), None)

if not filepath:
    print(f"Error: {filename} not found in current or parent directory.")
    exit(1)

print(f"Loading dataset from {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Total EO 13960 records: {len(df_eo)}")

# NOTE: Previous exploration showed '10_commercial_ai' contains use-case descriptions (e.g., 'Searching for information...'),
# not a binary Commercial/Custom flag. 
# Column '37_custom_code' contains 'Yes'/'No' values which is a better proxy for Commercial Opacity.
# 'No' Custom Code implies Commercial/COTS (Opaque).
# 'Yes' Custom Code implies Government/Custom (Transparent).

# 1. Define Commercial vs Custom based on '37_custom_code'
def categorize_source(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if s == 'no':
        return 'Commercial (COTS)'  # No custom code -> Commercial
    elif s == 'yes':
        return 'Custom (GOTS)'      # Custom code -> Government/Custom
    return np.nan

df_eo['source_category'] = df_eo['37_custom_code'].apply(categorize_source)

# Filter to valid rows
df_valid = df_eo.dropna(subset=['source_category']).copy()
print(f"Records with valid source category (Commercial/Custom): {len(df_valid)}")
print(df_valid['source_category'].value_counts())

# 2. Define Independent Evaluation Status
# '55_independent_eval' contains 'Yes...', 'TRUE', 'Planned', 'NaN', etc.
# Hypothesis requires 'having undergone', so 'Planned' is treated as 0.
def check_evaluation(val):
    if pd.isna(val):
        return 0
    s = str(val).lower().strip()
    # affirmative keywords
    if 'yes' in s or 'true' in s:
        return 1
    return 0

df_valid['has_eval'] = df_valid['55_independent_eval'].apply(check_evaluation)

# 3. Create Contingency Table
contingency = pd.crosstab(df_valid['source_category'], df_valid['has_eval'])
# Ensure columns are 0 and 1
if 0 not in contingency.columns:
    contingency[0] = 0
if 1 not in contingency.columns:
    contingency[1] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['No Eval', 'Has Eval']

print("\nContingency Table (Custom Code Status vs. Independent Eval):")
print(contingency)

# 4. Statistical Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 5. Odds Ratio
# Commercial (No Eval) = a, Commercial (Has Eval) = b
# Custom (No Eval) = c, Custom (Has Eval) = d
try:
    comm_row = contingency.loc['Commercial (COTS)']
    cust_row = contingency.loc['Custom (GOTS)']
    
    a = comm_row['No Eval']
    b = comm_row['Has Eval']
    c = cust_row['No Eval']
    d = cust_row['Has Eval']
    
    # Add smoothing if zeros exist
    if a*d == 0 or b*c == 0:
        print("\n(Using Haldane-Anscombe correction for zero cells)")
        odds_ratio = ((b + 0.5) * (c + 0.5)) / ((a + 0.5) * (d + 0.5))
    else:
        odds_ratio = (b * c) / (a * d)
        
    print(f"Odds Ratio (Commercial likelihood of Eval vs Custom): {odds_ratio:.4f}")
    
    # Interpret
    comm_rate = b / (a + b) * 100
    cust_rate = d / (c + d) * 100
    print(f"\nEvaluation Rate [Commercial/COTS]: {comm_rate:.2f}%")
    print(f"Evaluation Rate [Custom/GOTS]:     {cust_rate:.2f}%")
    
except KeyError as e:
    print(f"Error calculating odds ratio: {e}")
