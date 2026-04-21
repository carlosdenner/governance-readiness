import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting experiment: Sector-Specific Justice - Appeal Process Analysis")

# 1. Load Data
# Using the correct path verified in debug step
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored subset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 subset: {len(eo_data)} rows")

# 2. Map Agencies to Sectors
# Define sector mappings based on standard agency names
enforcement_agencies = [
    'Department of Homeland Security',
    'Department of Justice',
    'Department of State',
    'Department of Defense'
]

benefits_agencies = [
    'Department of Health and Human Services',
    'Department of Veterans Affairs',
    'Department of Education',
    'Department of Housing and Urban Development',
    'Social Security Administration'
]

def classify_sector(agency_name):
    if pd.isna(agency_name):
        return None
    # Normalize for matching
    name = str(agency_name).strip()
    if name in enforcement_agencies:
        return 'Enforcement'
    if name in benefits_agencies:
        return 'Benefits'
    return 'Other'

# Apply classification
eo_data['sector'] = eo_data['3_agency'].apply(classify_sector)

# Filter for only relevant sectors
analysis_df = eo_data[eo_data['sector'].isin(['Enforcement', 'Benefits'])].copy()
print(f"Filtered for target sectors: {len(analysis_df)} rows")
print(analysis_df['sector'].value_counts())

# 3. Parse '65_appeal_process'
# Inspect unique values to ensure robust parsing
unique_vals = analysis_df['65_appeal_process'].dropna().unique()
print(f"\nSample of raw '65_appeal_process' values (first 5): {unique_vals[:5]}")

def parse_appeal(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    # Heuristic: Check for explicit 'yes' or specific phrases indicating existence.
    if val_str.startswith('yes'):
        return 1
    return 0

analysis_df['has_appeal'] = analysis_df['65_appeal_process'].apply(parse_appeal)

# 4. Statistical Analysis
# Group by Sector
summary = analysis_df.groupby('sector')['has_appeal'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total Systems', 'With Appeal Process', 'Proportion']
print("\nSector Analysis Summary:")
print(summary)

# Prepare for Z-test
# Identify groups
sector_names = summary.index.tolist()
if len(sector_names) == 2:
    group1 = sector_names[0]
    group2 = sector_names[1]
    
    count1 = summary.loc[group1, 'With Appeal Process']
    nobs1 = summary.loc[group1, 'Total Systems']
    count2 = summary.loc[group2, 'With Appeal Process']
    nobs2 = summary.loc[group2, 'Total Systems']
    
    print(f"\nComparing: {group1} (n={nobs1}) vs {group2} (n={nobs2})")
    
    counts = np.array([count1, count2])
    nobs = np.array([nobs1, nobs2])
    
    stat, pval = proportions_ztest(counts, nobs)
    print(f"\nZ-Test Results:")
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value: {pval:.4f}")
    
    alpha = 0.05
    if pval < alpha:
        print("Conclusion: Significant difference in appeal process availability between sectors.")
    else:
        print("Conclusion: No significant difference found.")
else:
    print("Error: Insufficient groups for comparison.")

# 5. Visualization
plt.figure(figsize=(8, 6))
sns.barplot(x=summary.index, y='Proportion', data=summary.reset_index(), hue='sector', palette=['#1f77b4', '#ff7f0e'])
plt.title('Proportion of AI Systems with Defined Appeal Processes by Sector')
plt.ylabel('Proportion (0-1)')
plt.xlabel('Agency Sector')
plt.ylim(0, 1.0)

# Add labels
for i, row in enumerate(summary.itertuples()):
    # itertuples yields (Index, Total, With Appeal, Proportion)
    plt.text(i, row.Proportion + 0.02, f"{row.Proportion:.1%}\n(n={row._1})", 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
