import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from scipy.stats import chi2_contingency

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in parent directory
    file_path = '../astalabs_discovery_all_data.csv'
    df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Filtered EO 13960 records: {len(df_eo)}")

# --- Variable Creation ---

# Note: '10_commercial_ai' was found to contain use-case descriptions rather than 
# binary source indicators. Using '22_dev_method' as the robust proxy for 
# 'Development Source' (Government vs Vendor).

col_method = '22_dev_method'
col_impact = '52_impact_assessment'

# Define categorization logic for Development Source
def classify_source(val):
    if pd.isna(val):
        return None
    s = str(val).lower().strip()
    if 'in-house' in s and 'contracting' not in s:
        return 'Government'
    elif 'contracting' in s and 'in-house' not in s:
        return 'Commercial'
    # Exclude 'Both' or other ambiguous cases to ensure clean comparison
    return None

df_eo['Development_Source'] = df_eo[col_method].apply(classify_source)

# Define categorization logic for Impact Assessment
# Strict criteria: Only explicit 'Yes' counts as evidence.
def classify_impact(val):
    if pd.isna(val):
        return 0
    s = str(val).lower().strip()
    if s == 'yes':
        return 1
    return 0

df_eo['Has_Impact_Assessment'] = df_eo[col_impact].apply(classify_impact)

# Filter for valid analysis rows
df_analysis = df_eo.dropna(subset=['Development_Source'])
print(f"Records with valid Development Source: {len(df_analysis)}")
print(f"Distribution of Development Source:\n{df_analysis['Development_Source'].value_counts()}")

# --- Statistical Analysis ---

# Contingency Table
contingency = pd.crosstab(df_analysis['Development_Source'], df_analysis['Has_Impact_Assessment'])
contingency.columns = ['No Assessment', 'Has Assessment']
print("\nContingency Table (Source vs Impact Assessment):")
print(contingency)

# Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency)

# Compliance Rates
rates = df_analysis.groupby('Development_Source')['Has_Impact_Assessment'].mean() * 100
print("\nCompliance Rates (%):")
print(rates)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.5f}")

# Interpret
if p < 0.05:
    print("Result: Statistically Significant.")
    if rates['Commercial'] < rates['Government']:
        print("Hypothesis Supported: Commercial systems have significantly LOWER impact assessment rates.")
    else:
        print("Hypothesis Rejected: Commercial systems do NOT have lower rates.")
else:
    print("Result: Not Statistically Significant.")

# --- Visualization ---
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e']
ax = rates.plot(kind='bar', color=colors, alpha=0.8)
plt.title('Impact Assessment Compliance: Commercial vs. Government')
plt.ylabel('Compliance Rate (%)')
plt.xlabel('Development Source')
plt.xticks(rotation=0)
plt.ylim(0, max(rates.max() + 5, 10))

# Annotate bars
for i, v in enumerate(rates):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()