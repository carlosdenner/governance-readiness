import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug]
# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Normalize column names just in case, though they seem consistent based on previous output
# The relevant columns are 'Sector of Deployment' and 'Harm Distribution Basis'

# Clean and categorize Sector
def categorize_sector(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    if 'financial' in val_lower or 'finance' in val_lower or 'banking' in val_lower:
        return 'Financial'
    if 'transport' in val_lower or 'automotive' in val_lower:
        return 'Transportation'
    return None

aiid_df['target_sector'] = aiid_df['Sector of Deployment'].apply(categorize_sector)

# Filter for only the two sectors of interest
sector_df = aiid_df.dropna(subset=['target_sector']).copy()

# Define Demographic Harm
# Looking for 'Demographic', 'Race', 'Gender', etc. 
# Based on AIID taxonomy, 'Harm Distribution Basis' usually contains 'Demographic' for discrimination issues.
def is_demographic(val):
    if pd.isna(val):
        return False
    val_lower = str(val).lower()
    keywords = ['demographic', 'race', 'gender', 'sex', 'ethnicity', 'age', 'religion', 'disability']
    return any(k in val_lower for k in keywords)

sector_df['is_demographic_harm'] = sector_df['Harm Distribution Basis'].apply(is_demographic)

# Create Contingency Table
contingency_table = pd.crosstab(sector_df['target_sector'], sector_df['is_demographic_harm'])

# Rename columns for clarity in output
contingency_table.columns = ['Other Harm', 'Demographic Harm']

print("--- Contingency Table ---")
print(contingency_table)
print("\n")

# Check sample sizes for statistical test selection
total_samples = contingency_table.sum().sum()
min_expected = 0
if contingency_table.size == 4:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    min_expected = expected.min()
else:
    p = 1.0
    min_expected = 0

# Use Fisher's Exact Test if sample size is small or any expected cell count < 5
if min_expected < 5:
    print(f"Performing Fisher's Exact Test (Min expected count: {min_expected:.2f})...")
    oddsratio, p_value = stats.fisher_exact(contingency_table)
    test_name = "Fisher's Exact Test"
else:
    print(f"Performing Chi-Square Test (Min expected count: {min_expected:.2f})...")
    test_name = "Chi-Square Test"
    p_value = p

print(f"{test_name} Results:")
print(f"P-value: {p_value:.5f}")

# Calculate proportions
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print("\n--- Proportions ---")
print(props)

# Visualization
ax = props.plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'salmon'])
plt.title(f'Harm Distribution Basis by Sector\n({test_name} p={p_value:.4f})')
plt.ylabel('Proportion of Incidents')
plt.xlabel('Sector')
plt.legend(title='Harm Basis', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
