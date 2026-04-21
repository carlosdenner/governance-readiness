import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset shape: {subset.shape}")

# --- Process '16_dev_stage' for Operational Status ---
# Based on previous exploration, the values are:
# 'Operation and Maintenance', 'Acquisition and/or Development', 'Initiated',
# 'Implementation and Assessment', 'Retired', 'Planned', 'In production', 'In mission'

def check_operational(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    # Keywords to capture 'Operation', 'In production', 'In mission', 'Implementation'
    # The prompt asked for 'Implemented' or 'Operational', but the data uses variations.
    if any(keyword in s for keyword in ['operation', 'implement', 'production', 'mission']):
        return 1
    return 0

subset['is_operational'] = subset['16_dev_stage'].apply(check_operational)

# --- Process '52_impact_assessment' for Compliance ---
def check_impact(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 1
    return 0

subset['has_impact_assessment'] = subset['52_impact_assessment'].apply(check_impact)

# Print counts for verification
print("\nOperational Status Distribution:")
print(subset['is_operational'].value_counts())
print("\nImpact Assessment Distribution:")
print(subset['has_impact_assessment'].value_counts())

# --- Contingency Table ---
# Ensure we have a 2x2 table even if some categories are missing using reindex
contingency = pd.crosstab(subset['is_operational'], subset['has_impact_assessment'])
# Reindex to ensure all possibilities [0, 1] are present
contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

contingency.index = ['Pre-Operational', 'Operational']
contingency.columns = ['No Impact Assessment', 'Has Impact Assessment']

print("\n--- Contingency Table ---")
print(contingency)

# --- Statistical Analysis ---
chi2, p, dof, ex = chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio Calculation
a = contingency.loc['Pre-Operational', 'No Impact Assessment']
b = contingency.loc['Pre-Operational', 'Has Impact Assessment']
c = contingency.loc['Operational', 'No Impact Assessment']
d = contingency.loc['Operational', 'Has Impact Assessment']

if b * c > 0:
    odds_ratio = (d * a) / (c * b)
    print(f"Odds Ratio: {odds_ratio:.4f}")
else:
    print("Odds Ratio: Undefined (division by zero)")
