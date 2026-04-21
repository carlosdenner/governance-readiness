import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)} rows")

# Define target columns
col_autonomy = 'Autonomy Level'
col_intent = 'Intentional Harm'

# Check for missing data in target columns
subset = aiid[[col_autonomy, col_intent]].dropna()
print(f"Rows with complete data: {len(subset)}")

# Mapping functions based on observed values
def clean_autonomy(val):
    s = str(val).strip()
    if s == 'Autonomy1':
        return 'Low'
    elif s in ['Autonomy2', 'Autonomy3']:
        return 'High/Medium'
    return None  # Exclude 'unclear' or others

def clean_intent(val):
    s = str(val).strip()
    if s.startswith('Yes'):
        return 'Intentional'
    elif s.startswith('No'):
        return 'Unintentional'
    return None  # Exclude 'unclear' or others

# Apply mapping
subset['Autonomy_Bin'] = subset[col_autonomy].apply(clean_autonomy)
subset['Intent_Bin'] = subset[col_intent].apply(clean_intent)

# Drop unmapped rows
final_df = subset.dropna(subset=['Autonomy_Bin', 'Intent_Bin'])
print(f"Final analysis set: {len(final_df)} rows")

# Generate Contingency Table
# Hypothesis: Intentional -> Low Autonomy; Unintentional -> High Autonomy
contingency_table = pd.crosstab(final_df['Autonomy_Bin'], final_df['Intent_Bin'])

print("\n--- Contingency Table ---")
print(contingency_table)

# Perform Chi-square test
if contingency_table.size > 0 and contingency_table.sum().sum() > 0:
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("\n--- Chi-square Test Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Degrees of Freedom: {dof}")
    
    print("\n--- Expected Frequencies ---")
    print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

    # Interpretation
    alpha = 0.05
    if p < alpha:
        print("\nResult: Significant association found (reject H0).")
        # Check directionality by comparing observed vs expected
        # Specifically checking if Intentional is higher in Low Autonomy than expected
        obs_low_intent = contingency_table.loc['Low', 'Intentional'] if 'Low' in contingency_table.index and 'Intentional' in contingency_table.columns else 0
        exp_low_intent = expected[contingency_table.index.get_loc('Low'), contingency_table.columns.get_loc('Intentional')] if 'Low' in contingency_table.index and 'Intentional' in contingency_table.columns else 0
        
        if obs_low_intent > exp_low_intent:
            print("Direction: Low Autonomy is associated with Intentional Harm.")
        else:
            print("Direction: Association exists but direction differs from hypothesis.")
            
    else:
        print("\nResult: No significant association found (fail to reject H0).")
else:
    print("Contingency table is empty or invalid. Cannot perform Chi-square test.")