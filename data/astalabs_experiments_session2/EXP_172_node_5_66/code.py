import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Analysis of Impact vs SAOP Review ---")

# 1. Clean Independent Variable: Impact
# Mapping based on previous inspection:
# High Impact = 'Rights-Impacting', 'Safety-Impacting', 'Both'
# Low Impact = 'Neither'
def categorize_impact(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str in ['Rights-Impacting', 'Safety-Impacting', 'Safety-impacting', 'Both']:
        return 'High Impact'
    elif val_str == 'Neither':
        return 'Low Impact'
    else:
        return np.nan # Exclude NaN or unknown categories

eo_data['impact_binary'] = eo_data['17_impact_type'].apply(categorize_impact)

# 2. Clean Dependent Variable: SAOP Review
# Mapping: Yes/YES -> Yes, everything else (No, NaN, blank) -> No
def categorize_review(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).strip().upper()
    if val_str == 'YES':
        return 'Yes'
    return 'No'

eo_data['review_binary'] = eo_data['30_saop_review'].apply(categorize_review)

# Filter for valid impact data
valid_data = eo_data.dropna(subset=['impact_binary'])

# 3. Create Contingency Table
contingency_table = pd.crosstab(valid_data['impact_binary'], valid_data['review_binary'])

print("\nContingency Table (Impact [Rows] vs SAOP Review [Cols]):")
print(contingency_table)

# 4. Statistical Analysis
if 'Yes' in contingency_table.columns and 'High Impact' in contingency_table.index and 'Low Impact' in contingency_table.index:
    # Probability calculations
    high_row = contingency_table.loc['High Impact']
    low_row = contingency_table.loc['Low Impact']
    
    n_high = high_row.sum()
    k_high_yes = high_row['Yes'] if 'Yes' in high_row else 0
    prob_high = k_high_yes / n_high if n_high > 0 else 0
    
    n_low = low_row.sum()
    k_low_yes = low_row['Yes'] if 'Yes' in low_row else 0
    prob_low = k_low_yes / n_low if n_low > 0 else 0
    
    print(f"\nProbability of Review | High Impact: {prob_high:.2%} ({k_high_yes}/{n_high})")
    print(f"Probability of Review | Low Impact:  {prob_low:.2%} ({k_low_yes}/{n_low})")

    # Chi-square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Phi Coefficient
    n_total = contingency_table.sum().sum()
    phi = np.sqrt(chi2 / n_total)
    
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Phi Coefficient: {phi:.4f}")
    
    if p < 0.05:
        print("Result: Statistically Significant.")
        if prob_high > prob_low:
            print("Direction: Supports hypothesis (Higher impact leads to higher review probability).")
        else:
            print("Direction: Contradicts hypothesis.")
    else:
        print("Result: Not Statistically Significant.")
else:
    print("\nData structure insufficient for test (missing categories).")