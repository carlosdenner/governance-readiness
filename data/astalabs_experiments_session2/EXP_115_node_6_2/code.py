import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
else:
    filepath = filename

print(f"Loading data from {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# Filter AIID
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID rows: {len(aiid_df)}")

# Identify correct column names (handling potential lack of prefixes vs index from previous logs)
# Based on logs: 'Sector of Deployment' and 'Tangible Harm' should be the names
cols = aiid_df.columns.tolist()
sector_col = next((c for c in cols if 'Sector of Deployment' in c), None)
harm_col = next((c for c in cols if 'Tangible Harm' in c), None)

print(f"Using Sector Column: {sector_col}")
print(f"Using Harm Column: {harm_col}")

if not sector_col or not harm_col:
    print("Could not find required columns. Dumping available columns related to Sector or Harm:")
    print([c for c in cols if 'ector' in c or 'arm' in c])
    exit(1)

# Map sectors
def map_sector(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    if any(x in s for x in ['transportation', 'energy', 'construction']):
        return 'Physical_Infra'
    if any(x in s for x in ['finance', 'financial', 'education', 'admin', 'public administration']):
        return 'Services'
    return 'Other'

aiid_df['Sector_Group'] = aiid_df[sector_col].apply(map_sector)

# Filter groups
analysis_df = aiid_df[aiid_df['Sector_Group'].isin(['Physical_Infra', 'Services'])].copy()

# Map Harms
def check_physical_harm(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    if 'death' in s or 'injury' in s:
        return 1
    return 0

analysis_df['Is_Physical_Harm'] = analysis_df[harm_col].apply(check_physical_harm)

# Stats
rates = analysis_df.groupby('Sector_Group')['Is_Physical_Harm'].mean()
counts = analysis_df['Sector_Group'].value_counts()
print("\nRates of Physical Harm (Death/Injury):")
print(rates)
print("\nCounts:")
print(counts)

# Chi-square
ct = pd.crosstab(analysis_df['Sector_Group'], analysis_df['Is_Physical_Harm'])
print("\nContingency Table (0=No Harm, 1=Death/Injury):")
print(ct)
chi2, p, dof, ex = chi2_contingency(ct)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.6e}")

# Plot
plt.figure(figsize=(8,6))
# Define colors: Red for Physical/Infra (high risk assumed), Blue for Services
colors = []
for group in rates.index:
    if group == 'Physical_Infra':
        colors.append('#d9534f') # Red
    else:
        colors.append('#5bc0de') # Blue

bars = plt.bar(rates.index, rates.values, color=colors, alpha=0.8)
plt.title('Physical Harm Rate by Sector Group')
plt.ylabel('Rate of Death/Injury Incidents')
plt.xlabel('Sector Group')
plt.ylim(0, max(rates.values) * 1.2)

for bar, count in zip(bars, counts[rates.index]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height,
             f'{height:.1%} (n={count})', ha='center', va='bottom')

plt.tight_layout()
plt.show()