import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import sys

# Define file path (one level above as instructed)
file_path = '../astalabs_discovery_all_data.csv'

print("Loading dataset...")
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure during debug
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for the relevant source table
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {df_eo.shape}")

# Columns of interest
col_ato = '40_has_ato'
col_monitor = '56_monitor_postdeploy'
col_notice = '59_ai_notice'

# Function to map text to binary
# We need to inspect unique values to ensure accurate mapping, 
# but since we can't interactively check, we define robust keyword logic based on typical dataset patterns.

def map_binary(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    # Negative indicators
    if val_str in ['no', 'none', 'n/a', '0', 'false', 'not applicable', 'unknown']:
        return 0
    # If it has content that isn't explicitly negative, we assume affirmative for these fields
    # (e.g. "Yes", "Automated monitoring", "Specific notice provided")
    return 1

# Apply mapping
# Note: specific logic for 'monitor' from previous exploration context:
# "identifying positive monitoring indicators (e.g., 'automated', 'established')"
# We will use a slightly more nuanced mapper for each if necessary, but the general existence of text usually implies 'Yes' in this sparse dataset unless it says 'No'.

print("\n--- Unique Values Preview (Top 5) before mapping ---")
for col in [col_ato, col_monitor, col_notice]:
    print(f"{col}: {df_eo[col].unique()[:5]}")

# Refined Mapping Logic based on expected text
def parse_ato(val):
    s = str(val).lower()
    if 'yes' in s or 'true' in s or 'authorized' in s:
        return 1
    return 0

def parse_monitor(val):
    # Looking for affirmative descriptions
    s = str(val).lower()
    if pd.isna(val) or val == 'nan': return 0
    if s in ['no', 'none', 'n/a', 'not applicable']:
        return 0
    # If it contains description of a process, it's a Yes.
    return 1

def parse_notice(val):
    s = str(val).lower()
    if pd.isna(val) or val == 'nan': return 0
    if s in ['no', 'none', 'n/a', 'not applicable']:
        return 0
    return 1

df_eo['has_ato_bin'] = df_eo[col_ato].apply(parse_ato)
df_eo['monitor_bin'] = df_eo[col_monitor].apply(parse_monitor)
df_eo['notice_bin'] = df_eo[col_notice].apply(parse_notice)

# Helper to calculate stats
def calculate_association(df, col1, col2, label1, label2):
    cont_table = pd.crosstab(df[col1], df[col2])
    
    # If table is not 2x2, we pad it for consistent output, though mapping should ensure 0/1
    # We only run test if we have data
    if cont_table.size == 0:
        return None
        
    chi2, p, dof, ex = chi2_contingency(cont_table)
    n = cont_table.sum().sum()
    phi = np.sqrt(chi2 / n) if n > 0 else 0
    
    print(f"\nAnalysis: {label1} vs {label2}")
    print("Contingency Table:")
    print(cont_table)
    print(f"Chi-square: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    print(f"Phi Coefficient: {phi:.4f}")
    return phi, p

# 1. ATO vs Monitoring (Integration Readiness)
phi_ato_mon, p_ato_mon = calculate_association(df_eo, 'has_ato_bin', 'monitor_bin', 'ATO', 'Monitoring')

# 2. ATO vs Notice (Trust Readiness)
phi_ato_not, p_ato_not = calculate_association(df_eo, 'has_ato_bin', 'notice_bin', 'ATO', 'Notice')

print("\n--- Conclusion ---")
if p_ato_mon < 0.05 and phi_ato_mon > phi_ato_not:
    print("Result: Hypothesis SUPPORTED. ATO is more strongly correlated with Monitoring than with Notice.")
elif p_ato_mon < 0.05 and p_ato_not < 0.05 and phi_ato_mon <= phi_ato_not:
    print("Result: Hypothesis REJECTED. Correlation with Notice is stronger or equal.")
else:
    print("Result: Results inconclusive or lack statistical significance in one/both pairs.")
