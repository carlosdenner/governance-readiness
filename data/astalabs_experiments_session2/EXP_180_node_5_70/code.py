import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for the specific source table
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Broad Mapping of Public Service ---
# Assumption: Explicit descriptions are Public. 'No' or NaN are treated as Internal/Non-Public.
col_public = '26_public_service'

def map_service_type(val):
    s = str(val).strip().lower()
    # Known explicit public indicators from previous debug
    if s in ['nan', '', 'no']:
        return 'Internal/Non-Public'
    return 'Public'

eo_data['service_type'] = eo_data[col_public].apply(map_service_type)

# --- Step 2: Broad Mapping of Notice Compliance ---
col_notice = '59_ai_notice'

def map_notice_status(val):
    s = str(val).strip().lower()
    if s in ['nan', '']:
        return 'Unknown'
    
    # Explicit 'None of the above' is a failure to notify (Non-Compliant)
    if 'none of the above' in s:
        return 'No Notice'
    
    # Exemptions
    if any(x in s for x in ['n/a', 'waived', 'not safety', 'not interacting']):
        return 'Exempt'
    
    # Positive Indications
    if any(x in s for x in ['online', 'in-person', 'email', 'telephone', 'terms', 'instructions']):
        return 'Has Notice'
    
    return 'Unknown'

eo_data['notice_status'] = eo_data[col_notice].apply(map_notice_status)

# --- Step 3: Analysis Data ---
# We focus on rows where Notice is either 'Has Notice' or 'No Notice' (Binary Outcome).
analysis_df = eo_data[eo_data['notice_status'].isin(['Has Notice', 'No Notice'])].copy()

print("--- Data Distribution ---")
full_counts = pd.crosstab(eo_data['service_type'], eo_data['notice_status'])
print(full_counts)

print(f"\nAnalyzable Rows (Yes/No Notice): {len(analysis_df)}")

if len(analysis_df) < 5:
    print("Insufficient data to perform Chi-square test.")
else:
    # Generate Contingency Table for Test
    contingency = pd.crosstab(analysis_df['service_type'], analysis_df['notice_status'])
    print("\n--- Contingency Table (Analysis) ---")
    print(contingency)

    # Calculate Rates
    # Compliance Rate = Has Notice / (Has Notice + No Notice)
    # We calculate for both groups if they exist
    
    results = {}
    for group in contingency.index:
        has = contingency.loc[group, 'Has Notice'] if 'Has Notice' in contingency.columns else 0
        no = contingency.loc[group, 'No Notice'] if 'No Notice' in contingency.columns else 0
        total = has + no
        if total > 0:
            rate = (has / total) * 100
        else:
            rate = 0.0
        results[group] = {'rate': rate, 'total': total}

    print("\n--- Compliance Rates ---")
    for group, data in results.items():
        print(f"{group}: {data['rate']:.1f}% (n={data['total']})")

    # Chi-square Test
    # Only run if we have at least 2 groups and 2 outcomes in the full structure, 
    # but crosstab handles dimensions automatically. We need size > 0.
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(contingency)
        print("\n--- Chi-Square Results ---")
        print(f"Chi2: {chi2:.4f}")
        print(f"p-value: {p:.4e}")
        
        if p < 0.05:
            print("Result: Significant Difference.")
        else:
            print("Result: No Significant Difference.")
    else:
        print("\nCannot run Chi-square: One group or outcome is missing from the filtered data.")
