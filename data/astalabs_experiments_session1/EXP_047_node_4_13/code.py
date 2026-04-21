import pandas as pd
import scipy.stats as stats
import os

# Define filename
filename = 'step2_crosswalk_matrix.csv'

# Try loading from parent directory first, then current directory
filepath = os.path.join('..', filename)
if not os.path.exists(filepath):
    filepath = filename

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath)
    print(f"Successfully loaded. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Define the specific sources to compare based on previous output
regulatory_source = 'EU AI Act (2024/1689)'
technical_source = 'OWASP Top 10 LLM'

print(f"\n=== Hypothesis Testing: Regulatory ({regulatory_source}) vs Technical ({technical_source}) ===")

# Filter dataset
subset_df = df[df['source'].isin([regulatory_source, technical_source])]

# Create Contingency Table
contingency_table = pd.crosstab(subset_df['source'], subset_df['bundle'])
print("\nContingency Table:")
print(contingency_table)

# Check if table is 2x2
if contingency_table.shape == (2, 2):
    # Perform Fisher's Exact Test (appropriate for small sample sizes where cell counts < 5)
    # The table has a cell with value 1, so Fisher's is more robust than Chi-square here.
    odds_ratio, p_value_fisher = stats.fisher_exact(contingency_table)
    
    print(f"\nFisher's Exact Test:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value_fisher:.4f}")
    
    # Also performing Chi-Square for completeness as requested
    chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table, correction=True)
    print(f"\nChi-square Test (with Yates correction):")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p_value_chi2:.4f}")

    alpha = 0.05
    if p_value_fisher < alpha:
        print("\nResult: Significant difference found. The source framework strongly predicts the competency bundle.")
    else:
        print("\nResult: No statistically significant difference found (p >= 0.05).")
else:
    print("\nError: Contingency table is not 2x2. Check source names.")
    print("Found sources:", subset_df['source'].unique())