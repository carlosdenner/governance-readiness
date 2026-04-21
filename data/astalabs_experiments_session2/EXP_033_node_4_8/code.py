import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 records loaded: {len(eo_df)}")

# Step 1: Strict Agency Mapping
# Define explicit sets for Defense/Security
defense_abrs = {'DOD', 'DHS', 'DOJ', 'DOS', 'STATE'}
defense_names = {
    'Department of Defense',
    'Department of Homeland Security',
    'Department of Justice',
    'Department of State'
}

def categorize_agency_strict(row):
    abr = str(row.get('3_abr', '')).upper().strip()
    agency = str(row.get('3_agency', '')).strip()
    
    # Check exact abbreviations
    if abr in defense_abrs:
        return 'Defense/Security'
    
    # Check specific agency names (using startswith to catch sub-agencies if formatted like "Department of Defense - Army")
    # But based on previous output, names seem consistent. 
    # We strictly want to avoid "United States..." matching "State"
    for d_name in defense_names:
        if agency == d_name or agency.startswith(d_name + " "):
            return 'Defense/Security'
            
    return 'Civilian'

eo_df['agency_category'] = eo_df.apply(categorize_agency_strict, axis=1)

# Verify Classification
print("\n--- Verification of Defense/Security Agencies ---")
defense_agencies_found = eo_df[eo_df['agency_category'] == 'Defense/Security']['3_agency'].unique()
for ag in defense_agencies_found:
    print(f"  - {ag}")

# Step 2: Determine Procurement Type using '37_custom_code'
def categorize_procurement(val):
    val_str = str(val).lower().strip()
    if val_str == 'yes':
        return 'Custom (Build)'
    elif val_str == 'no':
        return 'Commercial (Buy)'
    else:
        return 'Unknown'

eo_df['procurement_type'] = eo_df['37_custom_code'].apply(categorize_procurement)

# Filter out Unknown procurement types
analysis_df = eo_df[eo_df['procurement_type'] != 'Unknown'].copy()

print(f"\nRecords after filtering for known procurement type: {len(analysis_df)}")
print(analysis_df['agency_category'].value_counts())

# Step 3: Contingency Table
contingency_table = pd.crosstab(analysis_df['agency_category'], analysis_df['procurement_type'])

print("\nContingency Table (Agency Category vs Procurement Type):")
print(contingency_table)

# Step 4: Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate percentages
props = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\nProportions (%):")
print(props)

# Step 5: Visualization
plt.figure(figsize=(10, 6))
ax = props.plot(kind='bar', stacked=True, color=['#ff7f0e', '#1f77b4'], figsize=(10, 6))

plt.title('AI Procurement Strategy: Defense/Security vs Civilian Agencies (Strict Mapping)')
plt.xlabel('Agency Sector')
plt.ylabel('Percentage of Systems')
plt.legend(title='Procurement Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)

# Add percentage labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white')

plt.tight_layout()
plt.show()