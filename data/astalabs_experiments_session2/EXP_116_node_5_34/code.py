import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os
import sys

# Load data
file_paths = ['../astalabs_discovery_all_data.csv', 'astalabs_discovery_all_data.csv']
df = None
for fp in file_paths:
    if os.path.exists(fp):
        print(f"Loading dataset from {fp}...")
        df = pd.read_csv(fp, low_memory=False)
        break

if df is None:
    print("Error: Dataset not found.")
    sys.exit(1)

# Filter for EO13960
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Total EO 13960 records: {len(eo_df)}")

# 1. Define High Impact
# Column: 17_impact_type
# Logic: contains 'rights' or 'safety' (case insensitive)
# Convert to string to handle NaNs safely
eo_df['17_impact_type_str'] = eo_df['17_impact_type'].fillna('').astype(str).str.lower()
eo_df['is_high_impact'] = eo_df['17_impact_type_str'].apply(lambda x: 1 if ('rights' in x or 'safety' in x) else 0)

# 2. Define Has Notice
# Column: 59_ai_notice
# Logic from prompt: Map to 0 if NaN, 'None of the above', or 'N/A'. Map to 1 otherwise.
def parse_notice(val):
    if pd.isna(val) or val == '':
        return 0
    s = str(val).lower().strip()
    if s == 'nan':
        return 0
    if 'none' in s:
        return 0
    if 'n/a' in s:
        return 0
    # If it's not None, not N/A, and not missing, we assume it describes a notice method
    return 1

eo_df['has_notice'] = eo_df['59_ai_notice'].apply(parse_notice)

# Debugging: show what values mapped to what
print("\nMapping check (Sample of values -> Result):")
sample_mapping = eo_df[['59_ai_notice', 'has_notice']].drop_duplicates().head(10)
print(sample_mapping)

# 3. Contingency Table
contingency = pd.crosstab(eo_df['is_high_impact'], eo_df['has_notice'])
contingency.index = ['Low Impact', 'High Impact']
contingency.columns = ['No Notice', 'Has Notice']

print("\nContingency Table:")
print(contingency)

# 4. Rates
low_stats = contingency.loc['Low Impact']
high_stats = contingency.loc['High Impact']

low_n = low_stats.sum()
high_n = high_stats.sum()

low_rate = low_stats['Has Notice'] / low_n if low_n > 0 else 0
high_rate = high_stats['Has Notice'] / high_n if high_n > 0 else 0

print(f"\nLow Impact Notice Rate:  {low_rate:.2%} ({low_stats['Has Notice']}/{low_n})")
print(f"High Impact Notice Rate: {high_rate:.2%} ({high_stats['Has Notice']}/{high_n})")

# 5. Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Conclusion: Significant difference in notice rates.")
else:
    print("Conclusion: No significant difference in notice rates.")