import pandas as pd
import scipy.stats as stats
import numpy as np

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Try one level up if not found in current directory as per hint, though instructions say "use dataset given"
    # Assuming standard path first based on previous context.
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

print("Dataset loaded successfully.")

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {df_eo.shape}")

# Normalize column names just in case (though previous output showed them clearly)
# Columns of interest: '29_contains_pii', '67_opt_out', '3_abr'

# Check unique values to ensure correct filtering
print("\nUnique values in '29_contains_pii':", df_eo['29_contains_pii'].unique())
print("Unique values in '67_opt_out':", df_eo['67_opt_out'].unique())

# Standardize values to boolean-like logic for analysis
# Assuming 'Yes' indicates presence. Adjusting for potential case sensitivity.
df_eo['has_pii'] = df_eo['29_contains_pii'].astype(str).str.strip().str.lower() == 'yes'
df_eo['has_opt_out'] = df_eo['67_opt_out'].astype(str).str.strip().str.lower() == 'yes'

# Filter for systems containing PII
df_pii = df_eo[df_eo['has_pii']].copy()
pii_count = len(df_pii)
print(f"\nSystems processing PII: {pii_count} (out of {len(df_eo)} total EO systems)")

if pii_count == 0:
    print("No PII systems found. Exiting analysis.")
else:
    # Calculate Privacy Control Gap (Percentage of PII systems missing Opt-Out)
    missing_opt_out_count = len(df_pii[~df_pii['has_opt_out']])
    privacy_control_gap = (missing_opt_out_count / pii_count) * 100

    print(f"\n--- Privacy-Control Gap Analysis ---")
    print(f"PII Systems with Opt-Out: {len(df_pii) - missing_opt_out_count}")
    print(f"PII Systems MISSING Opt-Out: {missing_opt_out_count}")
    print(f"Overall Privacy Control Gap: {privacy_control_gap:.2f}%")

    # Agency-level breakdown
    # Group by Agency ('3_abr')
    agency_col = '3_abr'
    
    # Create a crosstab of Agency vs Has_Opt_Out for PII systems only
    agency_stats = pd.crosstab(df_pii[agency_col], df_pii['has_opt_out'])
    agency_stats.columns = ['No_Opt_Out', 'Has_Opt_Out']  # False is No, True is Yes
    
    # Note: If all are True or all are False, crosstab might have 1 column. Handle this.
    if 'No_Opt_Out' not in agency_stats.columns:
        agency_stats['No_Opt_Out'] = 0
    if 'Has_Opt_Out' not in agency_stats.columns:
        agency_stats['Has_Opt_Out'] = 0
        
    agency_stats['Total_PII_Systems'] = agency_stats['No_Opt_Out'] + agency_stats['Has_Opt_Out']
    agency_stats['Gap_Percentage'] = (agency_stats['No_Opt_Out'] / agency_stats['Total_PII_Systems']) * 100
    
    # Filter for agencies with a meaningful number of PII systems (e.g., > 10) to reduce noise
    relevant_agencies = agency_stats[agency_stats['Total_PII_Systems'] >= 10].sort_values('Gap_Percentage', ascending=False)
    
    print("\n--- Agency-Level Breakdown (Agencies with >= 10 PII Systems) ---")
    print(relevant_agencies[['Total_PII_Systems', 'No_Opt_Out', 'Gap_Percentage']].round(2))

    # Chi-square test
    # We test if the distribution of Opt-Out (Yes/No) is independent of the Agency
    # Using the relevant agencies subset to ensure statistical validity
    if len(relevant_agencies) > 1:
        contingency_table = relevant_agencies[['No_Opt_Out', 'Has_Opt_Out']]
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\n--- Chi-Square Test for Independence (Agency vs Opt-Out Availability) ---")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")
        if p < 0.05:
            print("Result: Statistically significant. Opt-out availability depends on the agency.")
        else:
            print("Result: Not statistically significant. Opt-out availability appears independent of agency.")
    else:
        print("\nInsufficient data for valid Chi-square test across agencies.")
