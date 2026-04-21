import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Robust file loading
possible_paths = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    file_path = 'astalabs_discovery_all_data.csv' # Fallback

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 Scored dataset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {eo_data.shape}")

# 1. Define Biometric Logic
biometric_keywords = ['face', 'facial', 'biometric', 'recognition', 'fingerprint', 'voice', 'iris', 'dna', 'palm', 'gait']

def is_biometric(row):
    text = str(row.get('2_use_case_name', '')) + " " + str(row.get('11_purpose_benefits', ''))
    text = text.lower()
    return any(keyword in text for keyword in biometric_keywords)

eo_data['Is_Biometric'] = eo_data.apply(is_biometric, axis=1)

# 2. Define Mitigation Logic (Semantic search)
mitigation_col = '62_disparity_mitigation'
positive_keywords = ['test', 'eval', 'monitor', 'assess', 'audit', 'review', 'mitigat', 'bias', 'fair', 'human', 'check', 'guardrail', 'feedback', 'retrain', 'update']

def has_mitigation(row):
    text = str(row.get(mitigation_col, ''))
    if text.lower() == 'nan':
        return False
    # Check for positive indicators
    text_lower = text.lower()
    if any(pk in text_lower for pk in positive_keywords):
        return True
    return False

eo_data['Has_Mitigation'] = eo_data.apply(has_mitigation, axis=1)

# 3. Create Contingency Table & Ensure Dimensions
contingency_table = pd.crosstab(eo_data['Is_Biometric'], eo_data['Has_Mitigation'])

# Force 2x2 shape
contingency_table = contingency_table.reindex(index=[False, True], columns=[False, True], fill_value=0)

# Rename for clarity
contingency_table.index = ['Non-Biometric', 'Biometric']
contingency_table.columns = ['No Mitigation', 'Has Mitigation']

print("\nContingency Table (Counts):")
print(contingency_table)

# 4. Calculate Stats
biometric_total = contingency_table.loc['Biometric'].sum()
non_biometric_total = contingency_table.loc['Non-Biometric'].sum()

biometric_rate = contingency_table.loc['Biometric', 'Has Mitigation'] / biometric_total if biometric_total > 0 else 0
non_biometric_rate = contingency_table.loc['Non-Biometric', 'Has Mitigation'] / non_biometric_total if non_biometric_total > 0 else 0

print(f"\nBiometric Systems Mitigation Rate: {biometric_rate:.2%} (N={biometric_total})")
print(f"Non-Biometric Systems Mitigation Rate: {non_biometric_rate:.2%} (N={non_biometric_total})")

# 5. Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4f}")

# 6. Visualization
labels = ['Biometric', 'Non-Biometric']
rates = [biometric_rate, non_biometric_rate]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rates, color=['#d62728', '#1f77b4'], alpha=0.8)
plt.title('Disparity Mitigation Rates: Biometric vs. General AI Systems')
plt.ylabel('Proportion with Mitigation Measures')
plt.ylim(0, max(rates)*1.2 if max(rates) > 0 else 1.0)

# Add count labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    count = contingency_table.iloc[1-i, 1] # Biometric is index 1 (i=0), Non-Bio is index 0 (i=1)
    total = biometric_total if i == 0 else non_biometric_total
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1%}\n(n={count}/{total})', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()