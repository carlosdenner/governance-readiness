import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define dataset path
ds_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(ds_path):
    ds_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {ds_path}...")
try:
    df = pd.read_csv(ds_path, low_memory=False)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Filter for Federal Inventory (EO 13960)
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded {len(df_eo)} records from EO 13960 inventory.")

# Define relevant columns
col_dev_method = '22_dev_method'
col_docs = '34_data_docs'

# Check if columns exist
if col_dev_method not in df_eo.columns or col_docs not in df_eo.columns:
    print(f"Error: Required columns not found. Available columns: {df_eo.columns.tolist()}")
    exit(1)

# Mapping function for Development Method
# We treat 'Contracting resources' as the proxy for Vendor-Reliance/Commercial
# We treat 'In-house' as the proxy for Agency/Custom
def map_dev_method_refined(val):
    if pd.isna(val): return None
    val_str = str(val).lower()
    
    # Distinct categories based on the unique values found previously
    if 'contracting resources' in val_str and 'in-house' not in val_str:
        return 'Vendor/Contractor'
    
    if 'in-house' in val_str and 'contracting' not in val_str:
        return 'Agency/In-House'
        
    # Exclude hybrids ('both contracting and in-house') to ensure clean separation
    return None

# Mapping function for Documentation Compliance
def map_docs_compliance(val):
    if pd.isna(val): return None
    val_str = str(val).lower().strip()
    
    # Compliant categories
    if any(x in val_str for x in ['complete', 'widely available', 'yes', 'documentation is available']):
        return 'Compliant'
    
    # Non-Compliant categories (treating Partial as Non-Compliant for strict audit)
    if any(x in val_str for x in ['missing', 'no', 'partial', 'not available', 'none']):
        return 'Non-Compliant'
        
    return None

# Apply mappings
df_eo['System_Origin'] = df_eo[col_dev_method].apply(map_dev_method_refined)
df_eo['Compliance_Status'] = df_eo[col_docs].apply(map_docs_compliance)

# Filter for analysis
df_analysis = df_eo.dropna(subset=['System_Origin', 'Compliance_Status'])

print(f"\nRecords retained for analysis: {len(df_analysis)}")
print("Distribution of System Origins:")
print(df_analysis['System_Origin'].value_counts())

if len(df_analysis) < 10 or df_analysis['System_Origin'].nunique() < 2:
    print("Insufficient data/groups for Chi-Square test.")
else:
    # Create Contingency Table
    contingency = pd.crosstab(df_analysis['System_Origin'], df_analysis['Compliance_Status'])
    print("\n--- Contingency Table ---")
    print(contingency)

    # Calculate Rates
    rates = contingency.div(contingency.sum(axis=1), axis=0)
    print("\n--- Compliance Rates ---")
    print(rates * 100)
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n--- Chi-Square Test Results ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Interpretation
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    # Color mapping: Green for Compliant, Red for Non-Compliant
    colors = {'Compliant': '#2ca02c', 'Non-Compliant': '#d62728'}
    
    # Reorder columns if necessary to ensure stack order
    cols = [c for c in ['Non-Compliant', 'Compliant'] if c in rates.columns]
    rates_plot = rates[cols]
    plot_colors = [colors[c] for c in cols]
    
    rates_plot.plot(kind='bar', stacked=True, color=plot_colors, ax=plt.gca())
    plt.title('Data Documentation Compliance: Vendor vs Agency AI')
    plt.ylabel('Proportion')
    plt.xlabel('System Origin')
    plt.ylim(0, 1)
    plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
