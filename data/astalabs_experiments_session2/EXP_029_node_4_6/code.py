import pandas as pd
import numpy as np
import scipy.stats as stats

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
subset = df[df['source_table'] == 'eo13960_scored'].copy()

# Define column names
col_custom = '37_custom_code'
col_access = '38_code_access'

# Debug: Print unique values to define mapping logic
print(f"Unique values in '{col_custom}': {subset[col_custom].unique()}")
print(f"Unique values in '{col_access}': {subset[col_access].unique()}")

# Cleaning functions
def clean_commercial_proxy(val):
    # If 'custom code' is NO, we treat it as Commercial/COTS (Is Commercial = 1)
    # If 'custom code' is YES, we treat it as In-House/Custom (Is Commercial = 0)
    s = str(val).strip().lower()
    if 'no' in s:  # No custom code -> Commercial
        return 1
    elif 'yes' in s: # Yes custom code -> Non-Commercial
        return 0
    return np.nan

def clean_access(val):
    s = str(val).strip().lower()
    if 'no' in s: 
        return 0
    elif 'yes' in s:
        return 1
    return np.nan

# Apply cleaning
subset['is_commercial'] = subset[col_custom].apply(clean_commercial_proxy)
subset['has_code_access'] = subset[col_access].apply(clean_access)

# Drop NaNs for analysis
analysis_df = subset.dropna(subset=['is_commercial', 'has_code_access'])

print(f"\nData points after cleaning: {len(analysis_df)}")

if len(analysis_df) > 0:
    # Create Contingency Table
    # Rows: Commercial Status (0=Custom, 1=Commercial)
    # Cols: Code Access (0=No, 1=Yes)
    ct = pd.crosstab(analysis_df['is_commercial'], analysis_df['has_code_access'])
    print("\nContingency Table (Rows: Commercial [0=Custom, 1=Comm], Cols: Access [0=No, 1=Yes]):")
    print(ct)
    
    # Statistics
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    
    # Calculate Odds Ratio manually for 2x2: (a*d)/(b*c)
    # Table layout:
    #           Access=0   Access=1
    # Comm=0 (a)         (b)
    # Comm=1 (c)         (d)
    # OR of Access given Commercial? 
    # Let's calculate odds of *No Access* (Opacity) for Commercial vs Custom.
    # Opacity = Access 0.
    # Odds(Opacity | Commercial) = Count(Comm=1, Acc=0) / Count(Comm=1, Acc=1)
    # Odds(Opacity | Custom)     = Count(Comm=0, Acc=0) / Count(Comm=0, Acc=1)
    # OR = Odds(Comm) / Odds(Custom)
    
    try:
        # c = Comm=1, Acc=0; d = Comm=1, Acc=1
        # a = Comm=0, Acc=0; b = Comm=0, Acc=1
        c = ct.loc[1, 0] if 0 in ct.columns and 1 in ct.index else 0
        d = ct.loc[1, 1] if 1 in ct.columns and 1 in ct.index else 0
        a = ct.loc[0, 0] if 0 in ct.columns and 0 in ct.index else 0
        b = ct.loc[0, 1] if 1 in ct.columns and 0 in ct.index else 0
        
        odds_commercial_opacity = c / d if d > 0 else np.inf
        odds_custom_opacity = a / b if b > 0 else np.inf
        or_opacity = odds_commercial_opacity / odds_custom_opacity
        
        print(f"\nChi-Square p-value: {p:.4e}")
        print(f"Odds Ratio (Likelihood of Opacity for Commercial vs Custom): {or_opacity:.4f}")
        
        if p < 0.05:
            print("Significant result.")
            if or_opacity > 1:
                print("Commercial systems are significantly more opaque (less code access).")
            else:
                print("Commercial systems are significantly less opaque.")
        else:
            print("No significant difference found.")
            
    except Exception as e:
        print(f"Error calculating odds ratio: {e}")
        
else:
    print("No valid data points.")