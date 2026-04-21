import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset (trying current directory based on previous success)
filename = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filename, low_memory=False)
except FileNotFoundError:
    # Fallback to parent directory if current fails, though previous error suggests parent was wrong
    df = pd.read_csv('../' + filename, low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded {len(df_eo)} rows from EO 13960 source.")

# Construct Search Text
# Concatenate Topic Area and Use Case Name
df_eo['search_text'] = (df_eo['8_topic_area'].fillna('') + ' ' + df_eo['2_use_case_name'].fillna('')).str.lower()

# Define Keywords
high_stakes_kw = ['enforcement', 'security', 'surveillance', 'justice', 'health']
low_stakes_kw = ['admin', 'operations', 'management', 'logistics', 'finance']

# Vectorized categorization
def get_risk_level(text):
    if any(kw in text for kw in high_stakes_kw):
        return 'High Stakes'
    elif any(kw in text for kw in low_stakes_kw):
        return 'Low Stakes'
    return 'Other'

df_eo['risk_level'] = df_eo['search_text'].apply(get_risk_level)

# Categorize Impact Assessment
# Looking for explicit 'Yes'
df_eo['has_impact_assess'] = df_eo['52_impact_assessment'].fillna('No').astype(str).str.strip().str.lower() == 'yes'
df_eo['has_impact_assess_int'] = df_eo['has_impact_assess'].astype(int)

# Filter for analysis
df_analysis = df_eo[df_eo['risk_level'] != 'Other'].copy()

# Generate Contingency Table
contingency = pd.crosstab(df_analysis['risk_level'], df_analysis['has_impact_assess'])

print("\n--- Contingency Table (Risk Level vs Impact Assessment) ---")
print(contingency)

# Calculate Proportions
summary = df_analysis.groupby('risk_level')['has_impact_assess_int'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total Cases', 'With Assessment', 'Proportion']
print("\n--- Proportions ---")
print(summary)

# Statistical Test
if contingency.size > 0:
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Conclusion: Statistically significant difference exists.")
    else:
        print("Conclusion: No statistically significant difference found (Governance Decoupling supported).")
else:
    print("Insufficient data for Chi-square test.")
