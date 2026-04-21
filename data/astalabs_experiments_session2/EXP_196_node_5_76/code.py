import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the dataset
filepath = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filepath, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored records
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Validated column name from previous debug
mitigation_col = '62_disparity_mitigation'

# Clean Impact Type
def categorize_impact(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower().strip()
    
    # Strict separation: Exclude 'Both'
    if 'both' in val_str:
        return None
    
    is_safety = 'safety' in val_str
    is_rights = 'rights' in val_str
    
    if is_safety and not is_rights:
        return 'Safety-Impacted'
    elif is_rights and not is_safety:
        return 'Rights-Impacted'
    return None

eo_data['impact_category'] = eo_data['17_impact_type'].apply(categorize_impact)

# Filter for analysis
analysis_df = eo_data[eo_data['impact_category'].notna()].copy()

# Binarize Mitigation
def check_mitigation(val):
    if pd.isna(val):
        return 0
    # Looking for affirmative 'Yes'
    if str(val).lower().strip().startswith('yes'):
        return 1
    return 0

analysis_df['has_mitigation'] = analysis_df[mitigation_col].apply(check_mitigation)

# Create Contingency Table
contingency = pd.crosstab(analysis_df['impact_category'], analysis_df['has_mitigation'])
# Ensure we have both columns [0, 1]
contingency = contingency.reindex(columns=[0, 1], fill_value=0)
contingency.columns = ['No', 'Yes']

print("\nContingency Table (Impact Type vs. Disparity Mitigation):")
print(contingency)

# Calculate Rates
rates = analysis_df.groupby('impact_category')['has_mitigation'].mean()
print("\nMitigation Rates:")
print(rates)

# Statistical Test Check
total_positives = contingency['Yes'].sum()

if total_positives == 0:
    print("\nCannot perform Chi-Square test: No positive cases ('Yes') observed in either group.")
    print("Both groups have 0% compliance rate.")
    p_value = 1.0  # Technically undefined, but effectively no difference
else:
    # Only run test if we have data
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.6f}")
    except ValueError as e:
        print(f"\nStatistical test failed: {e}")

# Visualization
plt.figure(figsize=(8, 6))
colors = ['#e74c3c' if 'Safety' in idx else '#3498db' for idx in rates.index]
bars = plt.bar(rates.index, rates.values * 100, color=colors)

plt.title('Disparity Mitigation Controls: Safety vs. Rights Impacted Systems')
plt.ylabel('Percentage with Mitigation (%)')
plt.ylim(0, 100)

# Add annotations
for bar, idx in zip(bars, rates.index):
    height = bar.get_height()
    n_yes = contingency.loc[idx, 'Yes']
    n_total = contingency.loc[idx].sum()
    # If height is 0, place text slightly above 0
    y_pos = height + 1 if height > 0 else 2
    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
             f"{height:.1f}%\n(n={n_yes}/{n_total})", 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
