import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("Dataset not found. Please ensure 'astalabs_discovery_all_data.csv' is in the working directory.")
    exit(1)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Mapping Logic ---

def map_developer(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower()
    
    # Handle mixed/ambiguous cases first
    if 'both' in val_str:
        return 'Other'
        
    # Contractor keywords
    contractor_keys = ['contract', 'vendor', 'commercial', 'cots', 'third-party', 'external', 'purchase', 'industry']
    if any(k in val_str for k in contractor_keys):
        return 'Contractor'
    
    # In-house keywords
    inhouse_keys = ['government', 'agency', 'in-house', 'federal', 'staff', 'internal', 'developed by agency']
    if any(k in val_str for k in inhouse_keys):
        return 'In-house'
    
    return 'Other'

def map_documentation(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).lower()
    
    # Negative assertions (Priority 1)
    negative_keys = ['missing', 'not available', 'no documentation', 'not reported', 'none', 'unknown']
    if any(k in val_str for k in negative_keys):
        return 'No'
        
    # Positive assertions (Priority 2)
    positive_keys = ['complete', 'available', 'yes', 'link', 'http', 'partial', 'attached', 'pdf', 'doc']
    if any(k in val_str for k in positive_keys):
        return 'Yes'
        
    return 'No'

# Apply mappings
df_eo['Developer_Type'] = df_eo['22_dev_method'].apply(map_developer)
df_eo['Has_Docs'] = df_eo['34_data_docs'].apply(map_documentation)

# --- Validation ---
print("--- Data Mapping Validation (Sample) ---")
sample_docs = df_eo[['34_data_docs', 'Has_Docs']].drop_duplicates().dropna().head(10)
pd.set_option('display.max_colwidth', 100)
print(sample_docs)
print("\n")

# --- Analysis ---

# Filter for valid developer types
analysis_df = df_eo[df_eo['Developer_Type'].isin(['Contractor', 'In-house'])].copy()

# Contingency Table
contingency_table = pd.crosstab(analysis_df['Developer_Type'], analysis_df['Has_Docs'])

print("--- Contingency Table (Developer vs Documentation) ---")
print(contingency_table)

# Rates
rates = pd.crosstab(analysis_df['Developer_Type'], analysis_df['Has_Docs'], normalize='index') * 100
print("\n--- Documentation Rates (%) ---")
print(rates)

# Statistical Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-square Test Results ---")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant. The null hypothesis is rejected.")
else:
    print("Result: Not Statistically Significant. Failed to reject the null hypothesis.")

# Visualization
plt.figure(figsize=(10, 6))
# Re-calculate rates for plotting to ensure order
plot_data = rates[['Yes', 'No']] if 'Yes' in rates.columns and 'No' in rates.columns else rates
plot_data.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999'], ax=plt.gca())

plt.title('Data Documentation Availability by Developer Type (Refined)')
plt.xlabel('Developer Type')
plt.ylabel('Percentage')
plt.legend(title='Has Documentation', loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
