import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Prepare text fields for analysis (fill NaNs)
aiid_df['text_content'] = aiid_df['title'].fillna('') + " " + aiid_df['description'].fillna('')
aiid_df['text_content'] = aiid_df['text_content'].str.lower()

# --- MAPPING FUNCTIONS ---

def map_sector(s):
    s = str(s).lower()
    # Government/Public Sector
    if any(x in s for x in ['public administration', 'defense', 'law enforcement', 'justice', 'police', 'government', 'military']):
        return 'Government'
    # Healthcare
    if any(x in s for x in ['human health', 'medical', 'hospital', 'medicine', 'healthcare', 'clinical']):
        return 'Healthcare'
    return None

def map_harm_from_text(text):
    # Keywords for Rights/Liberties
    rights_keywords = ['discrimination', 'bias', 'privacy', 'surveillance', 'civil rights', 'liberty', 'racist', 'sexist', 'wrongful arrest', 'false arrest', 'denied', 'unfair']
    # Keywords for Physical Safety
    physical_keywords = ['death', 'killed', 'injury', 'injured', 'accident', 'crash', 'collision', 'physical harm', 'safety', 'died', 'fatal']
    
    has_rights = any(k in text for k in rights_keywords)
    has_physical = any(k in text for k in physical_keywords)
    
    if has_rights and not has_physical:
        return 'Rights/Liberties'
    if has_physical and not has_rights:
        return 'Physical/Safety'
    if has_rights and has_physical:
        return 'Mixed/Both' # Exclude to be clean
    return None

# Apply mappings
aiid_df['Sector_Group'] = aiid_df['Sector of Deployment'].apply(map_sector)
aiid_df['Harm_Group'] = aiid_df['text_content'].apply(map_harm_from_text)

# Filter for analysis (exclude Mixed or None)
final_df = aiid_df.dropna(subset=['Sector_Group', 'Harm_Group'])
final_df = final_df[final_df['Harm_Group'].isin(['Rights/Liberties', 'Physical/Safety'])]

print(f"Total AIID Incidents: {len(aiid_df)}")
print(f"Incidents with Sector/Harm data (inferred from text): {len(final_df)}")

if len(final_df) > 0:
    # Contingency Table
    contingency_table = pd.crosstab(final_df['Sector_Group'], final_df['Harm_Group'])
    print("\nContingency Table (Sector vs Harm):")
    print(contingency_table)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Row percentages
    row_pcts = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    print("\nRow Percentages:")
    print(row_pcts.round(2))
    
    # Check hypothesis
    if p < 0.05:
        gov_rights = row_pcts.loc['Government', 'Rights/Liberties'] if 'Rights/Liberties' in row_pcts.columns else 0
        health_phys = row_pcts.loc['Healthcare', 'Physical/Safety'] if 'Physical/Safety' in row_pcts.columns else 0
        
        print(f"\nGov Rights %: {gov_rights:.1f}%")
        print(f"Health Safety %: {health_phys:.1f}%")
        
        if gov_rights > 50 and health_phys > 50:
            print("\nRESULT: Hypothesis Supported. Government sector is dominated by Rights/Liberties harms, Healthcare by Physical/Safety harms.")
        else:
            print("\nRESULT: Significant difference found, but proportions may not fully align with hypothesis direction.")
    else:
        print("\nRESULT: No statistically significant association found.")
else:
    print("\nNo data available after text analysis mapping.")
