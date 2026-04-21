import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Define file path logic based on instruction
filename = 'astalabs_discovery_all_data.csv'
filepath = f'../{filename}' if os.path.exists(f'../{filename}') else filename

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath, low_memory=False)
except FileNotFoundError:
    print(f"Error: {filename} not found in current or parent directory.")
    exit(1)

# Filter for EO 13960 Scored subset
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {df_eo.shape}")

# Columns of interest
date_col = '18_date_initiated'
impact_col = '52_impact_assessment'

# Check if columns exist
if date_col not in df_eo.columns or impact_col not in df_eo.columns:
    print(f"Error: Missing columns. Available columns: {list(df_eo.columns)}")
    exit(1)

# Preview raw date data for debugging potential parsing issues
print("\nRaw date sample (first 5 non-null):")
print(df_eo[date_col].dropna().head(5).values)

# 1. Parse Dates
# Coerce errors to NaT (Not a Time) to handle malformed strings
df_eo['parsed_date'] = pd.to_datetime(df_eo[date_col], errors='coerce')

# Drop rows with missing or unparseable dates
df_clean = df_eo.dropna(subset=['parsed_date']).copy()
print(f"\nRows with valid dates: {len(df_clean)} (dropped {len(df_eo) - len(df_clean)} rows)")

# 2. Create Era Variable
# Extract year
df_clean['year'] = df_clean['parsed_date'].dt.year

# Define Era: Legacy (< 2021) vs Modern (>= 2021)
df_clean['Era'] = df_clean['year'].apply(lambda y: 'Legacy (<2021)' if y < 2021 else 'Modern (2021+)')

print("\nDistribution by Era:")
print(df_clean['Era'].value_counts())

# 3. Process Impact Assessment (Target)
# Normalize text: 'Yes' -> 1, Anything else -> 0
# Inspect unique values first
print(f"\nUnique values in '{impact_col}': {df_clean[impact_col].unique()}")

df_clean['Compliance'] = df_clean[impact_col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# 4. Statistical Analysis
# Contingency Table
contingency = pd.crosstab(df_clean['Era'], df_clean['Compliance'])
contingency.columns = ['Non-Compliant', 'Compliant']
print("\nContingency Table (Era vs. Compliance):")
print(contingency)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Test Results:")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate Compliance Rates
rates = df_clean.groupby('Era')['Compliance'].mean()
print("\nCompliance Rates (Proportion 'Yes'):")
print(rates)

# 5. Visualization
plt.figure(figsize=(8, 6))
ax = rates.plot(kind='bar', color=['#d95f02', '#1b9e77'], alpha=0.8, edgecolor='black')
plt.title('Impact Assessment Compliance: Legacy vs. Modern Systems')
plt.ylabel('Compliance Rate')
plt.xlabel('System Era')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=0)

# Add value labels
for i, v in enumerate(rates):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()