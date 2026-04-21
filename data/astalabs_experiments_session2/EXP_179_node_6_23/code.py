import pandas as pd
from scipy.stats import chi2_contingency
import os

# Define file path based on instructions
file_name = 'astalabs_discovery_all_data.csv'
file_path = f'../{file_name}'

if not os.path.exists(file_path):
    # Fallback to current directory if not found in parent
    if os.path.exists(file_name):
        file_path = file_name
    else:
        print(f"Error: {file_name} not found in current or parent directory.")
        exit(1)

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit(1)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {df_eo.shape}")

# Lifecycle Categorization
def categorize_stage(stage):
    if pd.isna(stage):
        return None
    stage_str = str(stage).lower()
    if 'operation' in stage_str or 'maintenance' in stage_str:
        return 'Operational'
    if 'development' in stage_str or 'implementation' in stage_str or 'planning' in stage_str:
        return 'Development'
    return None

df_eo['lifecycle_phase'] = df_eo['16_dev_stage'].apply(categorize_stage)

# Drop rows where phase could not be determined
df_eo = df_eo.dropna(subset=['lifecycle_phase'])

# Opt-Out Categorization
# Column '67_opt_out' is the target. Identify correct column name if slight variation exists.
target_col = '67_opt_out'
if target_col not in df_eo.columns:
    # Try to find it
    possible_cols = [c for c in df_eo.columns if 'opt_out' in c.lower()]
    if possible_cols:
        target_col = possible_cols[0]
        print(f"Warning: Exact column '{target_col}' found and used instead of '67_opt_out'.")
    else:
        print("Error: '67_opt_out' column not found.")
        exit(1)

# Create binary variable: 1 if 'Yes', 0 otherwise
df_eo['has_opt_out'] = df_eo[target_col].astype(str).str.strip().str.lower().apply(lambda x: 1 if x == 'yes' else 0)

# Generate Summary Statistics
summary = df_eo.groupby('lifecycle_phase')['has_opt_out'].agg(['count', 'sum', 'mean'])
summary['percent'] = summary['mean'] * 100

print("\n--- Opt-Out Documentation Rates by Lifecycle Phase ---")
print(summary)

# Contingency Table for Chi-Square
contingency_table = pd.crosstab(df_eo['lifecycle_phase'], df_eo['has_opt_out'])

# Ensure table has columns for both 0 and 1
for c in [0, 1]:
    if c not in contingency_table.columns:
        contingency_table[c] = 0
contingency_table = contingency_table[[0, 1]]

print("\n--- Contingency Table (0=No, 1=Yes) ---")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-Square Test Results ---")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Conclusion: Statistically significant difference detected.")
else:
    print("Conclusion: No statistically significant difference detected.")
