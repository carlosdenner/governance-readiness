import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filename, low_memory=False)
    print(f"Successfully loaded {filename}")
except FileNotFoundError:
    # Fallback if file is in parent directory
    df = pd.read_csv(f'../{filename}', low_memory=False)
    print(f"Successfully loaded ../{filename}")

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored Data Shape: {eo_data.shape}")

# --- 1. Define Commercial vs Custom ---
# Logic: 0 (Custom) if '10_commercial_ai' is 'None of the above.' or null.
#        1 (Commercial) otherwise.

def classify_commercial(val):
    s = str(val).strip()
    if pd.isna(val) or s.lower() == 'nan' or s == '':
        return 0
    if 'none of the above' in s.lower():
        return 0
    return 1

eo_data['is_commercial'] = eo_data['10_commercial_ai'].apply(classify_commercial)

print("\n--- Commercial Classification ---")
print(eo_data['is_commercial'].value_counts())
# Print a check to ensure we caught the specific commercial cases
print("Sample of Commercial descriptions:")
print(eo_data[eo_data['is_commercial'] == 1]['10_commercial_ai'].head(3).tolist())

# --- 2. Define Code Access (Target) ---
# Logic: 1 (Yes) if '38_code_access' starts with 'yes' (case insensitive).
#        0 (No) otherwise.

def classify_code_access(val):
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 1
    return 0

eo_data['has_code_access'] = eo_data['38_code_access'].apply(classify_code_access)

print("\n--- Code Access Classification ---")
print(eo_data['has_code_access'].value_counts())

# --- 3. Statistical Analysis ---
# Create contingency table
#              No Access (0)   Has Access (1)
# Custom (0)      n00             n01
# Commercial (1)  n10             n11
contingency_table = pd.crosstab(eo_data['is_commercial'], eo_data['has_code_access'])
contingency_table.index = ['Custom', 'Commercial']
contingency_table.columns = ['No Access', 'Has Access']

print("\n--- Contingency Table ---")
print(contingency_table)

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio Calculation
# OR = (Odds of Access for Custom) / (Odds of Access for Commercial)
#    = (n01 / n00) / (n11 / n10)

try:
    n00 = contingency_table.loc['Custom', 'No Access']
    n01 = contingency_table.loc['Custom', 'Has Access']
    n10 = contingency_table.loc['Commercial', 'No Access']
    n11 = contingency_table.loc['Commercial', 'Has Access']
    
    odds_custom = n01 / n00 if n00 != 0 else np.inf
    odds_commercial = n11 / n10 if n10 != 0 else np.inf
    
    print(f"\nOdds of Access (Custom): {odds_custom:.4f}")
    print(f"Odds of Access (Commercial): {odds_commercial:.4f}")

    if odds_commercial == 0:
        print("Odds Ratio undefined (Commercial odds is 0).")
    else:
        or_val = odds_custom / odds_commercial
        print(f"Odds Ratio (Custom vs Commercial): {or_val:.4f}")
        
        if p < 0.05:
            print("\nResult is statistically significant.")
            if or_val > 1:
                print(f"Custom systems are {or_val:.2f} times more likely to grant code access than Commercial systems.")
            else:
                print(f"Commercial systems are {1/or_val:.2f} times more likely to grant code access than Custom systems.")
        else:
            print("\nResult is not statistically significant.")

except Exception as e:
    print(f"Error calculating stats: {e}")
