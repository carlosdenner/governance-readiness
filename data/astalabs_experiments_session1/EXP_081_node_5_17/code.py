import pandas as pd
import scipy.stats as stats
import os

# Define file path based on instructions
file_path = '../step3_incident_coding.csv'

# Check if file exists at the expected location, else try current directory
if not os.path.exists(file_path):
    if os.path.exists('step3_incident_coding.csv'):
        file_path = 'step3_incident_coding.csv'
    else:
        print(f"Error: File not found at {file_path} or current directory.")
        exit(1)

print(f"Loading dataset from: {file_path}")

# Load the dataset
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Filter for 'security' and 'reliability' harm types
subset = df[df['harm_type'].isin(['security', 'reliability'])].copy()

print(f"Filtered subset size: {len(subset)} rows")
print(f"Harm Type counts:\n{subset['harm_type'].value_counts()}")
print(f"Failure Mode counts:\n{subset['failure_mode'].value_counts()}")

# Create binary classification: Prevention vs Other
# Note: 'Other' includes detection_failure, response_failure, etc.
subset['is_prevention'] = subset['failure_mode'].apply(lambda x: 1 if x == 'prevention_failure' else 0)

# Construct Contingency Table
# Rows: Security, Reliability
# Columns: Prevention (1), Other (0)

# Counts
sec_prev = len(subset[(subset['harm_type'] == 'security') & (subset['is_prevention'] == 1)])
sec_other = len(subset[(subset['harm_type'] == 'security') & (subset['is_prevention'] == 0)])

rel_prev = len(subset[(subset['harm_type'] == 'reliability') & (subset['is_prevention'] == 1)])
rel_other = len(subset[(subset['harm_type'] == 'reliability') & (subset['is_prevention'] == 0)])

# Table structure: [[Security_Prev, Security_Other], [Reliability_Prev, Reliability_Other]]
contingency_table = [[sec_prev, sec_other], [rel_prev, rel_other]]

print("\nContingency Table:")
print("              | Prevention | Other")
print(f"Security      | {sec_prev:<10} | {sec_other}")
print(f"Reliability   | {rel_prev:<10} | {rel_other}")

# Perform Fisher's Exact Test
# We use two-sided to test for any significant difference in distribution
odds_ratio, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')

print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("\nResult: Statistically significant difference found.")
else:
    print("\nResult: No statistically significant difference found.")
