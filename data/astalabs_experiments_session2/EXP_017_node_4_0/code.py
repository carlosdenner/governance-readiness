import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np
import os

# [debug] Check file existence to handle path variability
file_name = 'astalabs_discovery_all_data.csv'
file_path = f"../{file_name}" if os.path.exists(f"../{file_name}") else file_name

print(f"Loading dataset from: {file_path}")

# Load dataset
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Records: {len(eo_data)}")

# Define columns
col_opt_out = '67_opt_out'
col_appeal = '65_appeal_process'

# Inspect unique values for cleaning
print(f"Unique values in {col_opt_out}: {eo_data[col_opt_out].unique()}")
print(f"Unique values in {col_appeal}: {eo_data[col_appeal].unique()}")

# Data Cleaning Function
def clean_binary(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    # Check for affirmative values
    if val_str == 'yes' or val_str.startswith('yes'):
        return 1
    return 0

# Apply cleaning
eo_data['has_opt_out'] = eo_data[col_opt_out].apply(clean_binary)
neo_data = eo_data.copy() # Avoid SettingWithCopy warning on subsequent ops if any
neo_data['has_appeal'] = eo_data[col_appeal].apply(clean_binary)

# Create Contingency Table
ct = pd.crosstab(neo_data['has_opt_out'], neo_data['has_appeal'])
ct.index = ['No Opt-Out', 'Has Opt-Out']
ct.columns = ['No Appeal', 'Has Appeal']

print("\nContingency Table (Counts):")
print(ct)

# Calculate Percentages
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
print("\nContingency Table (Row Percentages):")
print(ct_pct)

# 1. Chi-Square Test
chi2, p, dof, expected = chi2_contingency(ct)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

# 2. Phi Coefficient (Correlation for binary variables)
# Phi = sqrt(chi2 / n)
n = ct.sum().sum()
phi = np.sqrt(chi2 / n)
print(f"Phi Coefficient (Correlation Strength): {phi:.4f}")

# 3. Jaccard Similarity (Intersection over Union for the 'Yes' condition)
# TP = Has Both, FP = Opt-Out Only, FN = Appeal Only
tp = ct.loc['Has Opt-Out', 'Has Appeal']
fp = ct.loc['Has Opt-Out', 'No Appeal']
fn = ct.loc['No Opt-Out', 'Has Appeal']
union = tp + fp + fn
jaccard = tp / union if union > 0 else 0.0
print(f"Jaccard Similarity Index (Overlap of 'Yes'): {jaccard:.4f}")

# Visualization: Stacked Bar Chart
plt.figure(figsize=(10, 6))
ax = ct_pct.plot(kind='bar', stacked=True, color=['#d62728', '#2ca02c'], alpha=0.8)
plt.title('Correlation between Opt-Out and Appeal Process (EO 13960)')
plt.xlabel('Opt-Out Mechanism Provided?')
plt.ylabel('Percentage of Systems')
plt.legend(title='Appeal Process Provided?', loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')

plt.tight_layout()
plt.show()
