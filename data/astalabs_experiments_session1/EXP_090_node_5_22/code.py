import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# [debug] Check if file exists in current or parent directory
file_name = 'step2_crosswalk_matrix.csv'
file_path = file_name
if not os.path.exists(file_path):
    if os.path.exists(f'../{file_name}'):
        file_path = f'../{file_name}'
    else:
        print(f"Error: {file_name} not found.")

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit(1)

# Map sources to types
# Technical: OWASP Top 10 LLM
# Normative: NIST AI RMF 1.0, NIST GenAI Profile, EU AI Act

def map_source(source_name):
    if 'OWASP' in str(source_name):
        return 'Technical'
    else:
        return 'Normative'

df['source_type'] = df['source'].apply(map_source)

# Create Contingency Table
contingency_table = pd.crosstab(df['source_type'], df['bundle'])

print("\n=== Contingency Table (Source Type vs Bundle) ===")
print(contingency_table)

# Calculate percentages for better context
contingency_pct = pd.crosstab(df['source_type'], df['bundle'], normalize='index') * 100
print("\n=== Contingency Table (Percentages) ===")
print(contingency_pct.round(1))

# Statistical Testing
# Using Fisher's Exact Test if 2x2, otherwise Chi-Square
chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Chi-Square Test of Independence ===")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p_val:.4f}")

# Calculate Cramer's V (Effect Size)
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Cramer's V: {cramers_v:.4f}")

# Fisher's Exact Test (specifically for 2x2 small samples, which this likely is)
if contingency_table.shape == (2, 2):
    # Note: scipy returns odds ratio and p-value
    odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
    print("\n=== Fisher's Exact Test (2x2) ===")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {fisher_p:.4f}")

# Interpretation
alpha = 0.05
print("\n=== Interpretation ===")
if p_val < alpha:
    print("Result: Statistically Significant Association.")
    print("There is a significant relationship between the source type (Normative vs Technical) and the competency bundle.")
else:
    print("Result: No Statistically Significant Association.")
    print("The source type does not appear to dictate the competency bundle significantly in this dataset.")
