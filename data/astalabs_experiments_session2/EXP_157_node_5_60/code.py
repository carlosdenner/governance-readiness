import pandas as pd
import scipy.stats as stats
import sys
import numpy as np

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 source table
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Identify columns dynamically
cols = df_eo.columns
topic_col = next((c for c in cols if 'topic_area' in c.lower()), None)
code_col = next((c for c in cols if 'code_access' in c.lower()), None)
data_col = next((c for c in cols if 'data_docs' in c.lower()), None)

if not all([topic_col, code_col, data_col]):
    print(f"Error: Critical columns missing. Found: Topic={topic_col}, Code={code_col}, Data={data_col}")
    sys.exit(1)

print(f"Processing with columns: '{topic_col}', '{code_col}', '{data_col}'")

# --- Step 1: Sector Categorization ---
# Ensure the column is string and handle NaNs
df_eo['topic_clean'] = df_eo[topic_col].fillna('').astype(str).str.lower()

# Define keywords
coercive_kw = ['law enforcement', 'defense', 'security', 'justice', 'border', 'police', 'military']
civil_kw = ['health', 'science', 'environment', 'education']

def classify_sector(val):
    if not isinstance(val, str):
        return 'Other'
    if any(k in val for k in coercive_kw):
        return 'Coercive'
    elif any(k in val for k in civil_kw):
        return 'Civil/Scientific'
    else:
        return 'Other'

df_eo['sector_group'] = df_eo['topic_clean'].apply(classify_sector)

# Filter dataset for analysis
df_analysis = df_eo[df_eo['sector_group'].isin(['Coercive', 'Civil/Scientific'])].copy()
print(f"Filtered for analysis: {len(df_analysis)} rows (Coercive + Civil/Scientific)")

# --- Step 2: Define Transparency ---
negative_values = ['no', 'none', 'not applicable', 'n/a', 'nan', 'false', '0', 'closed', 'restricted', '']

def is_transparent_feature(val):
    s = str(val).lower().strip()
    if s in negative_values or s == 'nan':
        return 0
    return 1

df_analysis['code_open'] = df_analysis[code_col].apply(is_transparent_feature)
df_analysis['data_open'] = df_analysis[data_col].apply(is_transparent_feature)

# Composite Variable: Transparent if EITHER code OR data docs are available
df_analysis['is_transparent'] = ((df_analysis['code_open'] == 1) | (df_analysis['data_open'] == 1)).astype(int)

# --- Step 3: Statistical Test ---
contingency = pd.crosstab(df_analysis['sector_group'], df_analysis['is_transparent'])

# Ensure both 0 and 1 columns exist
if 0 not in contingency.columns:
    contingency[0] = 0
if 1 not in contingency.columns:
    contingency[1] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['Opaque (0)', 'Transparent (1)']

print("\n--- Contingency Table (Sector vs Transparency) ---")
print(contingency)

# Calculate Transparency Rates
rates = df_analysis.groupby('sector_group')['is_transparent'].mean() * 100
print("\n--- Transparency Rates (% with Code or Data Access) ---")
print(rates)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print("\n--- Chi-Square Test Results ---")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p:.6f}")

if p < 0.05:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")
