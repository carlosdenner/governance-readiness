import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

# Load dataset
try:
    df = pd.read_csv('../step2_crosswalk_matrix.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('step2_crosswalk_matrix.csv')
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

print("=== Unique Sources in Dataset ===")
print(df['source'].unique())

# Map sources to categories based on hypothesis
# Policy: EU AI Act, NIST AI RMF
# Technical: OWASP, NIST GenAI Profile
def map_source(source_name):
    s = str(source_name).lower()
    if 'eu ai act' in s or 'nist ai rmf' in s:
        return 'Policy'
    elif 'owasp' in s or 'nist genai' in s:
        return 'Technical'
    else:
        return 'Uncategorized'

df['source_category'] = df['source'].apply(map_source)

# Remove any uncategorized if they exist (though expected to be 0)
df_clean = df[df['source_category'] != 'Uncategorized'].copy()

# Create Contingency Table
contingency_table = pd.crosstab(df_clean['source_category'], df_clean['bundle'])

print("\n=== Contingency Table (Source Category vs Competency Bundle) ===")
print(contingency_table)

# Perform Chi-Square Test of Independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Chi-Square Test Results ===")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p:.4f}")

# Calculate Cramer's V
# Formula: V = sqrt(chi2 / (n * (min(r, c) - 1)))
n = contingency_table.sum().sum()
r, c = contingency_table.shape
min_dim = min(r, c) - 1

if min_dim > 0 and n > 0:
    cramers_v = np.sqrt(chi2 / (n * min_dim))
else:
    cramers_v = 0.0

print(f"\n=== Effect Size ===")
print(f"Cramer's V: {cramers_v:.4f}")

# Interpretation
if p < 0.05:
    print("Conclusion: Statistically significant association detected.")
else:
    print("Conclusion: No statistically significant association detected.")

if cramers_v > 0.5:
    print("Strength: Strong")
elif cramers_v > 0.3:
    print("Strength: Medium")
elif cramers_v > 0.1:
    print("Strength: Small")
else:
    print("Strength: Negligible")