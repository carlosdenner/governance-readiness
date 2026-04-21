import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Define column names based on previous exploration
col_test = '53_real_world_testing'
col_eval = '55_independent_eval'

# strict mapping functions based on unique values
def map_testing(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    # 'operational environment' covers both 'Performance evaluation...' and 'Impact evaluation...'
    if 'operational environment' in s:
        return 1
    # Legacy/Simple boolean values
    if s in ['yes', 'true', '1']:
        return 1
    # 'Benchmark evaluation' explicitly states 'not been tested in an operational environment'
    return 0

def map_eval(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    # Check for explicit 'yes' or 'true'
    if s.startswith('yes') or s == 'true' or s == '1':
        return 1
    return 0

# Apply mapping
eo_df['has_testing'] = eo_df[col_test].apply(map_testing)
eo_df['has_eval'] = eo_df[col_eval].apply(map_eval)

# Generate Contingency Table
contingency = pd.crosstab(eo_df['has_testing'], eo_df['has_eval'])
print("Contingency Table (Rows: Real World Testing, Cols: Independent Eval):")
print(contingency)

# Stats
chi2, p, dof, ex = chi2_contingency(contingency)

# Phi Coefficient
n = contingency.sum().sum()
phi = np.sqrt(chi2 / n)

# Conditional Probability P(Eval | Testing)
# Testing=1 is the second row (index 1)
try:
    testing_yes_total = contingency.loc[1].sum()
    testing_yes_eval_yes = contingency.loc[1, 1]
    prob_eval_given_testing = testing_yes_eval_yes / testing_yes_total if testing_yes_total > 0 else 0
except KeyError:
    # Handle case where index 1 might not exist if no testing found
    prob_eval_given_testing = 0
    testing_yes_total = 0

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Phi Coefficient: {phi:.4f}")
print(f"Conditional Probability P(Independent Eval | Real World Testing): {prob_eval_given_testing:.2%}")
