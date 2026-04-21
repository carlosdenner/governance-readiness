import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# [debug] import matplotlib.pyplot as plt

print("Starting 'Opaque Shield' Experiment...\n")

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for EO 13960
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Dataset Loaded. Rows: {len(eo_df)}")

# 3. Categorize Agencies
# Defense/Security List per prompt: DOD, DHS, DOJ, VA, STATE (implied context of security/diplomacy/defense bundle)
# Civilian/Service List: HHS, ED, HUD, DOT, USDA, DOL, DOE, DOC, DOI, TREAS, SSA, EPA, GSA, NASA, NSF, NRC, OPM, SBA, USAID

defense_security_abbr = ['DOD', 'DHS', 'DOJ', 'VA', 'STATE']
civilian_service_abbr = ['HHS', 'ED', 'HUD', 'DOT', 'USDA', 'DOL', 'DOE', 'DOC', 'DOI', 'TREAS', 'SSA', 'EPA', 'GSA', 'NASA', 'NSF', 'NRC', 'OPM', 'SBA', 'USAID']

def categorize_agency(row):
    abr = str(row['3_abr']).upper().strip()
    agency = str(row['3_agency']).upper().strip()
    
    # Check exact abbreviation match first
    if abr in defense_security_abbr:
        return 'Defense/Security'
    if abr in civilian_service_abbr:
        return 'Civilian/Service'
        
    # Fallback to name keywords
    if any(x in agency for x in ['DEFENSE', 'HOMELAND', 'JUSTICE', 'VETERAN', 'STATE DEPARTMENT']):
        return 'Defense/Security'
    if any(x in agency for x in ['HEALTH', 'EDUCATION', 'HOUSING', 'TRANSPORTATION', 'AGRICULTURE', 'LABOR', 'ENERGY', 'COMMERCE', 'INTERIOR', 'TREASURY', 'SOCIAL SECURITY', 'ENVIRONMENTAL', 'AERONAUTICS', 'SCIENCE FOUNDATION']):
        return 'Civilian/Service'
    
    return 'Other/Unknown'

eo_df['agency_category'] = eo_df.apply(categorize_agency, axis=1)

# Filter for analysis
analysis_df = eo_df[eo_df['agency_category'] != 'Other/Unknown'].copy()

print("\nAgency Categorization Counts:")
print(analysis_df['agency_category'].value_counts())

# 4. Define Has_Notice
# Target: '59_ai_notice'. Parse categorical text.
# Positive: 'Online', 'Email', 'In-person', 'Yes', 'Physical', 'Mailed', 'Other'
# Negative: 'None', 'N/A', 'No', nan

def parse_notice(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    
    # Explicit negatives
    if 'none' in val_str or 'n/a' in val_str or val_str == 'no':
        return 0
        
    # Explicit positives or descriptive text indicating method
    positive_keywords = ['online', 'email', 'person', 'yes', 'physical', 'mail', 'other']
    if any(kw in val_str for kw in positive_keywords):
        return 1
        
    return 0

analysis_df['has_notice'] = analysis_df['59_ai_notice'].apply(parse_notice)

# 5. Analysis
group_stats = analysis_df.groupby('agency_category')['has_notice'].agg(['count', 'sum', 'mean'])
group_stats['percent'] = group_stats['mean'] * 100

contingency_table = pd.crosstab(analysis_df['agency_category'], analysis_df['has_notice'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

# 6. Deliverables
print("\n--- Analysis of Public Transparency (Notice) by Agency Type ---")
print(group_stats)
print("\nContingency Table (0=No Notice, 1=Has Notice):")
print(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Hypothesis Check
def_rate = group_stats.loc['Defense/Security', 'mean']
civ_rate = group_stats.loc['Civilian/Service', 'mean']

print("\n--- Conclusion ---")
if p < 0.05:
    if def_rate < civ_rate:
        print("Hypothesis SUPPORTED: Defense/Security agencies have significantly LOWER transparency rates.")
    else:
        print("Hypothesis REFUTED: Defense/Security agencies have significantly HIGHER transparency rates.")
else:
    print("Hypothesis NOT SUPPORTED: No significant difference found.")
