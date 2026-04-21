import pandas as pd
import numpy as np
from scipy import stats
import sys

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df_all = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

# Filter for eo13960_scored
df = df_all[df_all['source_table'] == 'eo13960_scored'].copy()

print(f"Loaded eo13960_scored with {len(df)} rows.")

# -- Inspect Development Method --
# We need to distinguish Commercial vs Custom.
# Metadata indicates '22_dev_method' is the relevant column.
col_dev_method = '22_dev_method'

if col_dev_method not in df.columns:
    print(f"Column '{col_dev_method}' not found. Available columns: {df.columns.tolist()}")
    sys.exit(1)

print(f"\nUnique values in '{col_dev_method}':")
print(df[col_dev_method].unique())

# Categorize Commercial vs Custom
def categorize_dev_method(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    
    # Commercial / COTS indicators
    if any(x in val_str for in ['cots', 'commercial', 'vendor', 'service', 'saas', 'bought']):
        return 'Commercial'
    
    # Custom / Government indicators
    if any(x in val_str for in ['custom', 'gots', 'government', 'internal', 'in-house', 'agency', 'developed']):
        return 'Custom'
        
    return np.nan

df['dev_category'] = df[col_dev_method].apply(categorize_dev_method)

# -- Inspect Documentation --
# Using '34_data_docs'
col_docs = '34_data_docs'

def categorize_docs(val):
    if pd.isna(val):
        return 0 # Treat NaN as No Docs
    val_str = str(val).lower().strip()
    
    # Affirmative keywords
    if any(x in val_str for in ['complete', 'partial', 'available', 'yes', 'public', 'source code']):
        return 1
    
    # Negative keywords (explicit)
    if any(x in val_str for in ['missing', 'not available', 'no', 'not reported']):
        return 0
        
    return 0 # Default to 0 if unclear

df['has_docs'] = df[col_docs].apply(categorize_docs)

# Filter for analysis
df_analysis = df.dropna(subset=['dev_category'])

print(f"\nRows with valid Dev Category: {len(df_analysis)}")
print(df_analysis['dev_category'].value_counts())

# Check groupings
group_counts = df_analysis.groupby(['dev_category', 'has_docs']).size().unstack(fill_value=0)
print("\n--- Group Counts ---")
print(group_counts)

# -- Statistical Test --
commercial_data = df_analysis[df_analysis['dev_category'] == 'Commercial']
custom_data = df_analysis[df_analysis['dev_category'] == 'Custom']

n_comm = len(commercial_data)
n_cust = len(custom_data)

if n_comm == 0 or n_cust == 0:
    print("\nInsufficient data in one or both groups.")
else:
    docs_comm = commercial_data['has_docs'].sum()
    docs_cust = custom_data['has_docs'].sum()
    
    prop_comm = docs_comm / n_comm
    prop_cust = docs_cust / n_cust
    
    print(f"\nCommercial Docs: {docs_comm}/{n_comm} ({prop_comm:.2%})")
    print(f"Custom Docs:     {docs_cust}/{n_cust} ({prop_cust:.2%})")
    
    # Contingency Table
    contingency = [[docs_comm, n_comm - docs_comm],
                   [docs_cust, n_cust - docs_cust]]
    
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    
    print(f"\nChi-square: {chi2:.4f}, p-value: {p:.4e}")
    
    if p < 0.05:
        if prop_comm < prop_cust:
            print("Hypothesis Supported: Commercial systems have significantly lower documentation rates.")
        else:
            print("Hypothesis Refuted: Commercial systems have higher documentation rates.")
    else:
        print("Result: No significant difference.")
