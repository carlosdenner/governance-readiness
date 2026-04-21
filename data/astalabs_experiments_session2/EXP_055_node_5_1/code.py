import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Initial EO 13960 records: {len(eo_data)}")

# --- Step 1: Define Agency Type (Security vs. Civilian) ---
# Security = Department of Defense, Homeland Security, Justice
col_agency = '3_agency'
if col_agency not in eo_data.columns:
    matches = [c for c in eo_data.columns if '3_' in c and 'agency' in c.lower()]
    if matches: col_agency = matches[0]

def categorize_agency(val):
    s = str(val).lower()
    if any(x in s for x in ['defense', 'homeland security', 'justice']):
        return 'Security/LE'
    return 'Civilian'

eo_data['agency_type'] = eo_data[col_agency].apply(categorize_agency)

print("\nAgency Type Distribution:")
print(eo_data['agency_type'].value_counts())

# Verify which agencies fell into Security
security_agencies = eo_data[eo_data['agency_type'] == 'Security/LE'][col_agency].unique()
print(f"\nAgencies classified as Security/LE: {list(security_agencies)[:5]}...")

# --- Step 2: Define Development Method (In-House vs. Outsourced) ---
col_dev = '22_dev_method'
if col_dev not in eo_data.columns:
    matches = [c for c in eo_data.columns if '22_' in c]
    if matches: col_dev = matches[0]

# Filter for clear-cut cases to ensure valid comparison
# Based on previous exploration, these are the two dominant clean categories
target_vals = ['Developed in-house.', 'Developed with contracting resources.']
analysis_df = eo_data[eo_data[col_dev].isin(target_vals)].copy()

def map_dev_method(val):
    if 'in-house' in str(val).lower():
        return 'In-House'
    return 'Outsourced'

analysis_df['dev_category'] = analysis_df[col_dev].apply(map_dev_method)

print(f"\nRecords for Analysis (Clean Dev Methods): {len(analysis_df)}")
print("Development Category Distribution:")
print(analysis_df['dev_category'].value_counts())

# --- Step 3: Statistical Test ---
contingency_table = pd.crosstab(analysis_df['agency_type'], analysis_df['dev_category'])
print("\nContingency Table (Rows=Agency, Cols=Method):")
print(contingency_table)

chi2, p_val, dof, expected = chi2_contingency(contingency_table)

# Calculate rates
rates = analysis_df.groupby('agency_type')['dev_category'].apply(lambda x: (x == 'In-House').mean())
sec_rate = rates.get('Security/LE', 0)
civ_rate = rates.get('Civilian', 0)

print(f"\nChi-Square Test Results:")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p_val:.4e}")
print(f"Security/LE In-House Rate: {sec_rate:.2%}")
print(f"Civilian In-House Rate: {civ_rate:.2%}")

# --- Step 4: Visualization ---
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e'] # Blue, Orange

# Plot In-House rates
bars = plt.bar(rates.index, rates.values, color=colors, alpha=0.8)

plt.title(f'"The Security Sovereignty of AI"\nIn-House Development Rates by Agency Mission (p={p_val:.1e})')
plt.ylabel('Proportion Developed In-House')
plt.ylim(0, 1.0)

# Add counts to bars
for bar, label in zip(bars, rates.index):
    height = bar.get_height()
    count = contingency_table.loc[label, 'In-House']
    total = contingency_table.loc[label].sum()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1%}\n(n={count}/{total})', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
