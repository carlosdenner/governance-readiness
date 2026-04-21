import pandas as pd
from scipy.stats import fisher_exact
import os

# 1. Load 'step3_incident_coding.csv'
file_path = '../step3_incident_coding.csv'
if not os.path.exists(file_path):
    # Fallback for local testing if needed
    file_path = 'step3_incident_coding.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

# 2. Filter for rows where 'trust_integration_split' is 'trust-dominant' or 'integration-dominant'
target_splits = ['trust-dominant', 'integration-dominant']
filtered_df = df[df['trust_integration_split'].isin(target_splits)].copy()

print(f"\nTotal incidents: {len(df)}")
print(f"Filtered incidents (excluding 'both'): {len(filtered_df)}")
print(f"Counts by split in filtered data:\n{filtered_df['trust_integration_split'].value_counts()}")

# 3. Create simplified failure mode variable: 'Prevention' vs 'Post-Prevention'
def simplify_failure(mode):
    if pd.isna(mode):
        return "Unknown"
    mode = str(mode).lower()
    if 'prevention' in mode:
        return 'Prevention'
    elif 'detection' in mode or 'response' in mode:
        return 'Post-Prevention'
    else:
        return 'Other'

filtered_df['failure_category'] = filtered_df['failure_mode'].apply(simplify_failure)

# 4. Create a contingency table of Failure Mode vs. Split
# We ensure the table structure is fixed for the test (2x2)
# Rows: integration-dominant, trust-dominant
# Cols: Prevention, Post-Prevention

expected_rows = ['integration-dominant', 'trust-dominant']
expected_cols = ['Prevention', 'Post-Prevention']

# Initialize matrix with zeros
test_matrix = pd.DataFrame(0, index=expected_rows, columns=expected_cols)

# Fill with actual counts
contingency = pd.crosstab(filtered_df['trust_integration_split'], filtered_df['failure_category'])

for r in expected_rows:
    for c in expected_cols:
        if r in contingency.index and c in contingency.columns:
            test_matrix.loc[r, c] = contingency.loc[r, c]

print("\nContingency Table (for Fisher's Exact Test):")
print(test_matrix)

# 5. Perform Fisher's Exact Test
# Note: If the sample size is very small or one category is empty, this handles it.
total_observations = test_matrix.sum().sum()
if total_observations < 2:
    print("\nInsufficient data to perform Fisher's Exact Test.")
else:
    odds_ratio, p_value = fisher_exact(test_matrix.to_numpy())
    print("\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio}")
    print(f"P-value: {p_value}")
    
    if p_value < 0.05:
        print("Result: Statistically significant association.")
    else:
        print("Result: No statistically significant association found (null hypothesis retained).")