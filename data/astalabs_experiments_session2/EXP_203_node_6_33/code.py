import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    # Using low_memory=False to handle mixed types warning from previous steps
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory (though instruction says one level above)
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# -- Step 1: Inspect and Clean Agency Data --
# Check unique agency abbreviations to ensure correct mapping
print("Unique Agency Abbreviations found:", eo_data['3_abr'].unique())

# Define Agency Categories based on hypothesis
# Safety-Critical: DHS, DOT, HHS, VA, DOD
# Administrative: DOC, TREASURY, GSA, SSA, HUD
# Mapping based on likely abbreviations found in federal datasets

safety_critical = ['DHS', 'DOT', 'HHS', 'VA', 'DOD', 'DOJ', 'DOE', 'STATE', 'USAID'] 
# Expanded slightly to include typical safety/nat-sec, but will focus on prompt's core list for strictness if needed.
# Let's stick strictly to the prompt's explicit examples + obvious ones if the abbreviation matches.
# Prompt examples: DHS, DOT, HHS, VA, DOD vs DOC, TREASURY, GSA, SSA, HUD.

safety_list = ['DHS', 'DOT', 'HHS', 'VA', 'DOD']
admin_list = ['DOC', 'TREASURY', 'GSA', 'SSA', 'HUD', 'ED', 'USDA', 'DOL'] # Added ED, USDA, DOL to admin as they are often policy/admin focused in this context, but let's stick to the prompt's specific list to test the specific hypothesis accurately.

# Re-defining strictly based on prompt to avoid confounding:
safety_target = ['DHS', 'DOT', 'HHS', 'VA', 'DOD']
admin_target = ['DOC', 'TREASURY', 'GSA', 'SSA', 'HUD']

def categorize_agency(abr):
    if abr in safety_target:
        return 'Safety-Critical'
    elif abr in admin_target:
        return 'Administrative'
    else:
        return None

eo_data['agency_type'] = eo_data['3_abr'].apply(categorize_agency)

# Filter out uncategorized agencies
analysis_df = eo_data.dropna(subset=['agency_type']).copy()

print(f"\nRows after filtering for target agencies: {len(analysis_df)}")
print(analysis_df['agency_type'].value_counts())

# -- Step 2: Clean Impact Assessment Data --
# Check values in '52_impact_assessment'
col_impact = '52_impact_assessment'
print(f"\nUnique values in {col_impact}:", analysis_df[col_impact].unique())

# Convert to binary. Assuming 'Yes'/'No' or '1'/'0' or boolean.
# Standardizing to string for inspection then mapping
analysis_df['impact_bool'] = analysis_df[col_impact].astype(str).str.lower().map({'yes': 1, 'true': 1, '1': 1, '1.0': 1, 'no': 0, 'false': 0, '0': 0, '0.0': 0})

# Check for NaNs after mapping
print(f"NaNs in impact_bool after mapping: {analysis_df['impact_bool'].isna().sum()}")
# Fill NaNs with 0 if safe (assuming missing = no assessment), but better to drop if unsure. 
# Given EO inventories often have 'No' explicit, let's see. If NaN is high, we might assume 0.
# For rigorous stats, we drop NaNs or assume 0. Let's assume 0 as 'Not Reported' usually equals 'None' in compliance.
analysis_df['impact_bool'] = analysis_df['impact_bool'].fillna(0)

# -- Step 3: Statistical Analysis --
# Create contingency table
contingency = pd.crosstab(analysis_df['agency_type'], analysis_df['impact_bool'])
print("\nContingency Table (0=No, 1=Yes):")
print(contingency)

# Chi-Square Test
chi2, p, dof, ex = chi2_contingency(contingency)
print(f"\nChi-Square Test Results:\nStatistic: {chi2:.4f}, p-value: {p:.4e}")

# Calculate rates
rates = analysis_df.groupby('agency_type')['impact_bool'].mean()
print("\nImpact Assessment Adoption Rates:")
print(rates)

# -- Step 4: Visualization --
plt.figure(figsize=(8, 6))
ax = rates.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Adoption of Impact Assessments: Safety-Critical vs Administrative')
plt.ylabel('Proportion of AI Systems with Impact Assessment')
plt.xlabel('Agency Mission Type')
plt.ylim(0, 1.0)

# Add value labels
for i, v in enumerate(rates):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()