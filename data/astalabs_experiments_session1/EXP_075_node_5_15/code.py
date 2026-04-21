import pandas as pd
import scipy.stats as stats
import os

# Define file path
file_name = 'step3_incident_coding.csv'
file_path = file_name

# Robust path checking based on feedback
if not os.path.exists(file_path):
    # Check parent directory just in case
    if os.path.exists(f"../{file_name}"):
        file_path = f"../{file_name}"
    else:
        print(f"Error: {file_name} not found in current or parent directory.")
        # List current dir for debugging purposes if file not found
        print("Current directory contents:", os.listdir('.'))
        exit(1)

print(f"Loading {file_path}...")
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit(1)

# Data Processing
# 1. Harm Category: Security vs Other
df['harm_normalized'] = df['harm_type'].astype(str).str.strip().str.lower()
df['Harm_Category'] = df['harm_normalized'].apply(lambda x: 'Security' if x == 'security' else 'Other')

# 2. Failure Category: Prevention vs Detection/Response
df['failure_normalized'] = df['failure_mode'].astype(str).str.strip().str.lower()

def classify_failure(val):
    if 'prevention' in val:
        return 'Prevention'
    elif 'detection' in val or 'response' in val:
        return 'Detection/Response'
    else:
        return 'Other/Unknown'

df['Failure_Category'] = df['failure_normalized'].apply(classify_failure)

# Filter out Unknown if any
df_clean = df[df['Failure_Category'] != 'Other/Unknown'].copy()

# Generate Contingency Table
# Rows: Harm (Security, Other)
# Cols: Failure (Prevention, Detection/Response)
contingency = pd.crosstab(df_clean['Harm_Category'], df_clean['Failure_Category'])

# Ensure all columns/rows exist for the test
expected_cols = ['Prevention', 'Detection/Response']
for col in expected_cols:
    if col not in contingency.columns:
        contingency[col] = 0

# Reorder columns
contingency = contingency[expected_cols]

# Ensure all rows exist
expected_rows = ['Security', 'Other']
for row in expected_rows:
    if row not in contingency.index:
        contingency.loc[row] = [0, 0]

# Reorder rows
contingency = contingency.reindex(expected_rows)

print("\n=== Contingency Table ===")
print(contingency)

# Fisher's Exact Test
odds_ratio, p_value = stats.fisher_exact(contingency)

print("\n=== Fisher's Exact Test Results ===")
print(f"Odds Ratio: {odds_ratio}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Result: Significant association (p < 0.05)")
else:
    print("Result: No significant association (p >= 0.05)")

# Insight into the rare class
print("\n=== Detailed Breakdown of Non-Prevention Failures ===")
non_prev = df[df['Failure_Category'] == 'Detection/Response']
if not non_prev.empty:
    print(non_prev[['case_study_id', 'harm_type', 'failure_mode']])
else:
    print("No Detection/Response failures found.")