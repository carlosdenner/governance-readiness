import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define file path
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
elif os.path.exists(filename):
    filepath = filename
else:
    filepath = filename 
    print("File not found in expected locations, attempting current dir...")

# Load dataset
df = pd.read_csv(filepath, low_memory=False)

# Filter for EO 13960
target_source = 'eo13960_scored'
eo_df = df[df['source_table'] == target_source].copy()
print(f"Filtered for {target_source}: {len(eo_df)} records")

# Columns
col_method = '22_dev_method'
col_mitig = '62_disparity_mitigation'

# Map Procurement Method (Commercial vs Custom)
def map_procurement(val):
    s = str(val).strip().lower()
    if 'contracting' in s and 'in-house' not in s:
        return 'Commercial (Vendor)'
    elif 'in-house' in s and 'contracting' not in s:
        return 'Custom (In-House)'
    else:
        return np.nan # Exclude Mixed or Unknown for clean comparison

eo_df['procurement_type'] = eo_df[col_method].apply(map_procurement)

# Map Mitigation Status (Documented vs Not)
def map_mitigation(val):
    if pd.isna(val):
        return 'Not Documented'
    s = str(val).strip().lower()
    # Check for non-substantive answers
    if s in ['nan', 'none', 'n/a', 'no', 'not applicable', 'none.', 'na']:
        return 'Not Documented'
    if len(s) < 5: # arbitrarily short strings likely meaningless
        return 'Not Documented'
    return 'Documented'

eo_df['mitigation_status'] = eo_df[col_mitig].apply(map_mitigation)

# Filter for analysis
analysis_df = eo_df.dropna(subset=['procurement_type'])

print(f"Records for analysis (Commercial vs Custom): {len(analysis_df)}")
print("Breakdown by Procurement Type:")
print(analysis_df['procurement_type'].value_counts())

# Contingency Table
ct = pd.crosstab(analysis_df['procurement_type'], analysis_df['mitigation_status'])
print("\nContingency Table (Count):")
print(ct)

# Percentages
ct_pct = pd.crosstab(analysis_df['procurement_type'], analysis_df['mitigation_status'], normalize='index') * 100
print("\nContingency Table (Percentage):")
print(ct_pct)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-Square Test Results:")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Plot
if 'Documented' in ct_pct.columns:
    rates = ct_pct['Documented']
else:
    rates = pd.Series([0, 0], index=ct.index)

plt.figure(figsize=(8, 6))
colors = ['#ff9999' if 'Commercial' in x else '#66b3ff' for x in rates.index]
bars = plt.bar(rates.index, rates.values, color=colors, edgecolor='black')
plt.title('Disparity Mitigation Documentation Rate: Commercial vs Custom AI')
plt.ylabel('Percent with Documented Mitigation (%)')
plt.xlabel('Procurement Method')
plt.ylim(0, max(rates.values)*1.2 if len(rates)>0 and max(rates.values)>0 else 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()