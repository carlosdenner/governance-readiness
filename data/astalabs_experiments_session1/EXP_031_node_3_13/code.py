import pandas as pd
import scipy.stats as stats
import os

# --- robust file loading ---
filename = 'step3_incident_coding.csv'
file_paths = [filename, f'../{filename}']

df = None
for path in file_paths:
    if os.path.exists(path):
        print(f"Found dataset at: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    print(f"Error: Could not find {filename} in current or parent directory.")
    # List current directory contents for debugging if needed, but per instructions, we exit.
    exit(1)

# --- Data Preparation ---
print("\n=== Dataset Statistics ===")
print(f"Total rows: {len(df)}")
print("\nDistribution of 'trust_integration_split':")
print(df['trust_integration_split'].value_counts())
print("\nDistribution of 'failure_mode':")
print(df['failure_mode'].value_counts())

# Filter out 'Both' to focus on distinct dominance
# We normalize to lowercase to be safe, though metadata suggests 'Both' is capitalized
subset = df[~df['trust_integration_split'].str.lower().eq('both')].copy()
print(f"\nSubset size (excluding 'Both'): {len(subset)}")

if len(subset) == 0:
    print("No records found with distinct Trust or Integration dominance.")
    exit(0)

# Map failure modes
# Prevention vs Post-Incident (Detection/Response)
subset['failure_category'] = subset['failure_mode'].apply(
    lambda x: 'Prevention' if 'prevention' in str(x).lower() else 'Post-Incident'
)

print("\nSubset Failure Category Distribution:")
print(subset['failure_category'].value_counts())

# --- Contingency Table ---
contingency_table = pd.crosstab(subset['trust_integration_split'], subset['failure_category'])
print("\n=== Contingency Table (Split vs Failure Category) ===")
print(contingency_table)

# --- Statistical Test ---
# We need at least 2 rows and 2 columns to test independence between variables.
# If all rows fall into one column (e.g., all Prevention), we cannot run Chi-Square/Fisher.

rows, cols = contingency_table.shape
if rows >= 2 and cols >= 2:
    if rows == 2 and cols == 2:
        odds_ratio, p_value = stats.fisher_exact(contingency_table)
        print(f"\nFisher's Exact Test Results:")
        print(f"Odds Ratio: {odds_ratio}")
        print(f"P-value: {p_value}")
    else:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-Square Test Results:")
        print(f"Chi2 Stat: {chi2}")
        print(f"P-value: {p}")
else:
    print("\nSkipping statistical test: Contingency table dimensions are insufficient (need at least 2x2).")
    print("Observed dimensions:", (rows, cols))
    print("This indicates a lack of variance in one or both variables within the subset.")