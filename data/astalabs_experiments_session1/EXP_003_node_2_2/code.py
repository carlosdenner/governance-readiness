import pandas as pd
from scipy.stats import fisher_exact
import sys

# Define the file path based on the note provided
file_path = '../step3_coverage_map.csv'

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback to current directory if ../ fails, just in case, though note specified ../
    print("File not found at ../, trying current directory...")
    df = pd.read_csv('step3_coverage_map.csv')

# 2. Create a binary flag 'is_uncovered' (incident_count == 0)
df['is_uncovered'] = df['incident_count'] == 0

# 3. Group by 'bundle' and count
# First, let's see the raw counts
print("\nTotal Sub-competencies per Bundle:")
print(df['bundle'].value_counts())

print("\nUncovered Sub-competencies per Bundle (Count of True):")
print(df.groupby('bundle')['is_uncovered'].sum())

# Create contingency table for Fisher's Exact Test
# format: pd.crosstab(index, columns)
# Rows: Bundle, Columns: Is Uncovered
contingency_table = pd.crosstab(df['bundle'], df['is_uncovered'])
print("\nContingency Table (Rows: Bundle, Cols: Is Uncovered):")
print(contingency_table)

# 4. Perform Fisher's Exact Test
# The hypothesis implies we want to see if one bundle is *more* likely to be uncovered.
# Fisher's exact test is suitable for small sample sizes (n=16).
odds_ratio, p_value = fisher_exact(contingency_table)

print("\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("\nResult: The difference in coverage gaps between bundles IS statistically significant.")
else:
    print("\nResult: The difference in coverage gaps between bundles IS NOT statistically significant.")
