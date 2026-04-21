import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure, though instructions say one level above
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Loaded EO 13960 dataset with {len(eo_df)} rows.")

# --- Preprocessing ---

# Function to binarize 'Yes'/'No' text responses
def binarize_response(text):
    if pd.isna(text):
        return 0
    # Check if the string starts with 'yes' (case-insensitive) to capture verbose responses
    if str(text).strip().lower().startswith('yes'):
        return 1
    return 0

# Target columns
col_ato = '40_has_ato'
col_eval = '55_independent_eval'

# Binarize
eo_df['has_ato_bin'] = eo_df[col_ato].apply(binarize_response)
eo_df['has_eval_bin'] = eo_df[col_eval].apply(binarize_response)

# Print value counts to verify parsing
print(f"\nDistribution of ATO (Security) Compliance:\n{eo_df['has_ato_bin'].value_counts()}")
print(f"\nDistribution of Independent Eval (Safety) Compliance:\n{eo_df['has_eval_bin'].value_counts()}")

# --- Statistical Analysis ---

# Create Contingency Table
contingency_table = pd.crosstab(eo_df['has_ato_bin'], eo_df['has_eval_bin'])
contingency_table.index = ['No ATO', 'Has ATO']
contingency_table.columns = ['No Indep. Eval', 'Has Indep. Eval']

print("\n--- Contingency Table ---")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Calculate Phi Coefficient (Mean Square Contingency Coefficient for 2x2)
# Phi = sqrt(chi2 / n)
n = contingency_table.sum().sum()
phi_coefficient = np.sqrt(chi2 / n)

# Determine sign of association by comparing observed vs expected for the (1,1) cell
# If observed (Has ATO, Has Eval) > expected, positive association.
obs_yes_yes = contingency_table.loc['Has ATO', 'Has Indep. Eval']
exp_yes_yes = expected[1, 1]
if obs_yes_yes < exp_yes_yes:
    phi_coefficient = -phi_coefficient

print("\n--- Statistical Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Phi Coefficient: {phi_coefficient:.4f}")

# Calculate Odds Ratio for interpretation
# (Has ATO & Has Eval / Has ATO & No Eval) / (No ATO & Has Eval / No ATO & No Eval)
try:
    a = contingency_table.loc['Has ATO', 'Has Indep. Eval']
    b = contingency_table.loc['Has ATO', 'No Indep. Eval']
    c = contingency_table.loc['No ATO', 'Has Indep. Eval']
    d = contingency_table.loc['No ATO', 'No Indep. Eval']
    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else np.inf
    print(f"Odds Ratio: {odds_ratio:.4f}")
except Exception as e:
    print(f"Could not calculate Odds Ratio: {e}")

# --- Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Security (ATO) vs Safety (Independent Eval) Compliance')
plt.ylabel('Security Compliance (ATO)')
plt.xlabel('Safety Compliance (Independent Eval)')
plt.show()

# --- Interpretation ---
print("\n--- Interpretation ---")
if p < 0.05:
    print("There is a statistically significant relationship between ATO and Independent Evaluation.")
    if phi_coefficient > 0:
        print("The relationship is POSITIVE: Systems with ATO are more likely to have Independent Evaluation.")
    else:
        print("The relationship is NEGATIVE: Systems with ATO are less likely to have Independent Evaluation.")
else:
    print("There is NO statistically significant relationship. Security compliance (ATO) does not predict Safety compliance.")
