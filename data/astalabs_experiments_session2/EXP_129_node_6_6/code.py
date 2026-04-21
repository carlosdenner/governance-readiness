import pandas as pd
import scipy.stats as stats
import sys

# [debug] verify execution environment
print("Starting experiment: Mission Culture (HHS vs DHS) Disparity Mitigation")

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Normalize agency names
eo_data['3_agency'] = eo_data['3_agency'].astype(str).str.strip()

# Define target agencies (Switching DOD to DHS based on data availability)
target_map = {
    'Department of Health and Human Services': 'HHS',
    'Department of Homeland Security': 'DHS'
}

# Filter data
study_df = eo_data[eo_data['3_agency'].isin(target_map.keys())].copy()
study_df['target_agency'] = study_df['3_agency'].map(target_map)

print(f"Filtered dataset size: {len(study_df)} rows")
print(study_df['target_agency'].value_counts())

if len(study_df['target_agency'].unique()) < 2:
    print("Error: Insufficient agencies found for comparison.")
    sys.exit(0)

# Analyze '62_disparity_mitigation' column
raw_mitigation_col = '62_disparity_mitigation'

def classify_mitigation(val):
    if pd.isna(val):
        return False
    text = str(val).lower().strip()
    
    # Negative indicators
    negative_terms = ['no', 'n/a', 'none', 'not applicable', 'not assessed', 'tbd', 'unknown']
    if text in negative_terms:
        return False
    if any(text.startswith(term) for term in negative_terms):
        # Check for cases like "No, ..." but allow "Note..."
        if text.startswith('no ') or text.startswith('no,') or text.startswith('no.'):
            return False
        if text.startswith('n/a'):
            return False
            
    # Positive indicators
    positive_terms = ['yes', 'mitigat', 'assess', 'review', 'test', 'monitor', 'evaluat', 'analyz', 'audit', 'bias']
    if any(term in text for term in positive_terms):
        return True
    
    # If text is substantial but not explicitly negative, treat as potential positive? 
    # Sticking to conservative heuristic: needs positive keyword.
    return False

study_df['has_mitigation'] = study_df[raw_mitigation_col].apply(classify_mitigation)

# Create Contingency Table
contingency_table = pd.crosstab(study_df['target_agency'], study_df['has_mitigation'])

# Ensure all columns exist and order is [False, True]
# Using reindex with fill_value=0 handles missing columns safely
contingency_table = contingency_table.reindex(columns=[False, True], fill_value=0)

# Rename columns for clarity
contingency_table.columns = ['No Mitigation', 'Has Mitigation']

print("\n--- Contingency Table (Agency vs Disparity Mitigation) ---")
print(contingency_table)

# Calculate Rates
rates = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\n--- Mitigation Rates (% of Agency Systems) ---")
print(rates)

# Statistical Test (Chi-Square)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- Statistical Test Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant Difference")
else:
    print("Result: No Statistically Significant Difference")

# Calculate Odds Ratio (HHS vs DHS)
# OR = (HHS_Yes / HHS_No) / (DHS_Yes / DHS_No)
try:
    hhs_yes = contingency_table.loc['HHS', 'Has Mitigation']
    hhs_no = contingency_table.loc['HHS', 'No Mitigation']
    dhs_yes = contingency_table.loc['DHS', 'Has Mitigation']
    dhs_no = contingency_table.loc['DHS', 'No Mitigation']
    
    # Add small epsilon to avoid division by zero if needed, though Fisher's test usually handles this better
    # Just simple calculation here
    odds_hhs = hhs_yes / hhs_no if hhs_no > 0 else float('inf')
    odds_dhs = dhs_yes / dhs_no if dhs_no > 0 else float('inf')
    
    if odds_dhs > 0 and odds_dhs != float('inf'):
        odds_ratio = odds_hhs / odds_dhs
        print(f"Odds Ratio (HHS relative to DHS): {odds_ratio:.4f}")
    elif odds_dhs == 0:
         print("Odds Ratio: Undefined (DHS has 0 odds)")
    else:
         print("Odds Ratio: Undefined (Infinite odds)")
         
except KeyError:
    print("Could not calculate Odds Ratio due to missing agency in index.")
