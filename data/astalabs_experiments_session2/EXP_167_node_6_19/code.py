import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Robust Data Loading ---
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, f'../{filename}']
ds_path = None

for path in possible_paths:
    if os.path.exists(path):
        ds_path = path
        break

if ds_path is None:
    raise FileNotFoundError(f"{filename} not found in current or parent directory.")

print(f"Loading dataset from {ds_path}...")
df = pd.read_csv(ds_path, low_memory=False)

# --- 2. Filter Data ---
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {eo_data.shape}")

# --- 3. Binarization ---
col_impact = '52_impact_assessment'
col_notice = '59_ai_notice'

# Logic for Impact Assessment (Control 52)
# We treat 'Yes' as 1. 'Planned' is treated as 0 (not yet compliant).
def binarize_impact(val):
    if pd.isna(val):
        return 0
    s_val = str(val).lower().strip()
    if s_val in ['yes', 'true', '1', '1.0']:
        return 1
    return 0

# Logic for AI Notice (Control 59)
# Based on observed values, we filter out explicit non-compliance/exclusions.
def binarize_notice(val):
    if pd.isna(val):
        return 0
    s_val = str(val).strip()
    
    # explicit negatives
    negatives = [
        'None of the above',
        'N/A - individuals are not interacting with the AI for this use case',
        'Agency CAIO has waived this minimum practice and reported such waiver to OMB.',
        'AI is not safety or rights-impacting.'
    ]
    if s_val in negatives:
        return 0
    
    # If it's not a negative and is a non-empty string, it's a form of notice (Online, Email, etc.)
    if len(s_val) > 0:
        return 1
    return 0

eo_data['impact_binary'] = eo_data[col_impact].apply(binarize_impact)
eo_data['notice_binary'] = eo_data[col_notice].apply(binarize_notice)

print(f"\nBinary counts for Impact Assessment (1=Yes):\n{eo_data['impact_binary'].value_counts()}")
print(f"Binary counts for AI Notice (1=Provided):\n{eo_data['notice_binary'].value_counts()}")

# --- 4. Create Contingency Table ---
contingency_table = pd.crosstab(eo_data['impact_binary'], eo_data['notice_binary'])
# Ensure 2x2
contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

# Display Table
display_table = contingency_table.copy()
display_table.index = ['No Impact Assessment', 'Has Impact Assessment']
display_table.columns = ['No AI Notice', 'Has AI Notice']

print("\n--- Contingency Table ---")
print(display_table)

# --- 5. Statistical Analysis ---
# Check for degeneracy (if any row/col sums to 0)
row_sums = contingency_table.sum(axis=1)
col_sums = contingency_table.sum(axis=0)

if (row_sums == 0).any() or (col_sums == 0).any():
    print("\nWarning: Degenerate contingency table (one or more categories have 0 observations).")
    print("Cannot perform Chi-square test.")
else:
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Phi Coefficient
    n = contingency_table.sum().sum()
    phi = np.sqrt(chi2 / n)

    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Phi Coefficient: {phi:.4f}")

    # --- 6. Visualization ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(display_table, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Co-occurrence: Impact Assessment vs. AI Notice')
    plt.ylabel('Impact Assessment (Control 52)')
    plt.xlabel('AI Notice (Control 59)')
    plt.tight_layout()
    plt.show()

    # --- 7. Interpretation ---
    alpha = 0.05
    print("\n--- Conclusion ---")
    if p < alpha:
        print("Result: Statistically Significant Dependency.")
        if phi > 0.5:
            strength = "Strong"
        elif phi > 0.3:
            strength = "Moderate"
        elif phi > 0.1:
            strength = "Weak"
        else:
            strength = "Negligible"
        print(f"There is a {strength} positive association (Phi={phi:.2f}) between the controls.")
        print("Hypothesis Supported: Agencies that perform Impact Assessments are significantly more likely to provide AI Notice.")
    else:
        print("Result: No Statistically Significant Dependency.")
        print("Hypothesis Rejected: Governance controls do not appear to cluster significantly in this dataset.")
