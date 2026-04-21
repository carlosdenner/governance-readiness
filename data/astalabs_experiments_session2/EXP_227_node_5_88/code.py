import pandas as pd
import scipy.stats as stats
import sys

print("Starting Commercial Obscurity hypothesis test (Attempt 2)...\n")

# 1. Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for 'eo13960_scored'
df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded {len(df)} rows from EO 13960 dataset.")

# 2. Define Variables and Mappings
col_dev_method = '22_dev_method'
col_code_access = '38_code_access'

# Check columns
if col_dev_method not in df.columns or col_code_access not in df.columns:
    print(f"Error: Columns '{col_dev_method}' or '{col_code_access}' missing.")
    sys.exit(1)

# Mapping for Procurement (Independent Variable)
def map_procurement(val):
    s = str(val).strip()
    if pd.isna(val) or s.lower() == 'nan':
        return None
    if s == 'Developed with contracting resources.':
        return 'Commercial'
    elif s == 'Developed in-house.':
        return 'Custom/Internal'
    # Exclude 'Developed with both...' and 'Data not reported...' to ensure clean groups
    return None

# Mapping for Code Access (Dependent Variable)
# 'Transparent' = Publicly Available
# 'Opaque' = No access, or Internal access only (not public)
def map_transparency(val):
    s = str(val).strip()
    if pd.isna(val) or s.lower() == 'nan':
        return None # Missing data, exclude
    
    # Strict check for Public availability
    if 'publicly available' in s.lower():
        return 'Transparent'
    
    # All other non-missing values are Opaque (Restricted/Closed)
    # Includes: 'No – agency does not have access...', 'Yes – agency has access... but it is not public', 'Yes', 'YES'
    return 'Opaque'

# Apply mappings
df['Procurement_Type'] = df[col_dev_method].apply(map_procurement)
df['Code_Transparency'] = df[col_code_access].apply(map_transparency)

# Filter for valid rows (where both fields are not None)
df_analysis = df.dropna(subset=['Procurement_Type', 'Code_Transparency']).copy()

print(f"\nAnalysis set size after filtering: {len(df_analysis)}")

# 3. Generate Statistics

# Contingency Table
contingency_table = pd.crosstab(df_analysis['Procurement_Type'], df_analysis['Code_Transparency'])

print("\n--- Contingency Table ---")
print(contingency_table)

# Percentages
row_sums = contingency_table.sum(axis=1)
percentages = contingency_table.div(row_sums, axis=0) * 100

print("\n--- Percentages (Row-wise) ---")
print(percentages.round(2))

# 4. Statistical Test (Chi-Square)
if contingency_table.size > 0 and contingency_table.sum().sum() > 0:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    print(f"Degrees of freedom: {dof}")
    
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically SIGNIFICANT difference found.")
        
        # Check directionality
        comm_trans = percentages.loc['Commercial', 'Transparent'] if 'Transparent' in percentages.columns else 0
        cust_trans = percentages.loc['Custom/Internal', 'Transparent'] if 'Transparent' in percentages.columns else 0
        
        print(f"Commercial Transparency: {comm_trans:.2f}%")
        print(f"Custom/Internal Transparency: {cust_trans:.2f}%")
        
        if comm_trans < cust_trans:
            print("The data SUPPORTS the hypothesis: Commercial systems are less transparent.")
        else:
            print("The data CONTRADICTS the hypothesis direction.")
    else:
        print("\nResult: No statistically significant difference found (Fail to reject Null).")
else:
    print("\nInsufficient data for Chi-Square test.")
