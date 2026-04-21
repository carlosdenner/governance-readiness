import pandas as pd
import scipy.stats as stats
import numpy as np

# Load the dataset
filename = 'step2_crosswalk_matrix.csv'
df = pd.read_csv(filename)

# Inspect and clean the 'source' column
# We normalize by stripping whitespace to handle potential formatting issues
df['source'] = df['source'].astype(str).str.strip()

# Define the target groups using robust string matching
# This handles cases where the string might be 'EU AI Act ' or similar variants
def classify_source(s):
    if 'OWASP' in s:
        return 'OWASP Top 10 LLM'
    elif 'EU AI' in s:
        return 'EU AI Act'
    else:
        return None

# Apply classification
df['target_source'] = df['source'].apply(classify_source)

# Filter for only the relevant rows
sub_df = df[df['target_source'].notna()].copy()

print("--- Data Inspection ---")
print(f"Unique sources found in raw data: {df['source'].unique()}")
print(f"Filtered dataset shape: {sub_df.shape}")
print("Counts per target source:")
print(sub_df['target_source'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(sub_df['target_source'], sub_df['bundle'])
print("\n--- Contingency Table ---")
print(contingency_table)

# Perform Statistical Test
# We use Fisher's Exact Test if the table is 2x2, as sample sizes are likely small
if contingency_table.shape == (2, 2):
    test_name = "Fisher's Exact Test"
    # fisher_exact returns (odds_ratio, p_value)
    statistic, p_value = stats.fisher_exact(contingency_table)
    stat_label = "Odds Ratio"
else:
    test_name = "Chi-Square Test"
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    statistic = chi2
    stat_label = "Chi-Square Statistic"

print(f"\n--- Statistical Analysis ({test_name}) ---")
print(f"{stat_label}: {statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpretation
alpha = 0.05
print("\n--- Interpretation ---")
if p_value < alpha:
    print(f"Result: Significant (p < {alpha}). The source framework predicts the readiness bundle.")
else:
    print(f"Result: Not Significant (p >= {alpha}). No statistical evidence that source predicts bundle.")