import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
import os
import sys

# 1. Load Data
filename = 'astalabs_discovery_all_data.csv'
file_path = filename if os.path.exists(filename) else os.path.join('..', filename)

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: {filename} not found.")
    sys.exit(1)

# 2. Filter for ATLAS cases / Incident Coding
target_sources = ['atlas_cases', 'step3_incident_coding']
atlas_df = df[df['source_table'].isin(target_sources)].copy()
print(f"Filtered {len(atlas_df)} records from sources: {target_sources}")

# 3. Identify Tactics Column
# Priority: 'tactics', then 'tactics_used', avoiding 'n_tactics'
tactics_col = None
if 'tactics' in atlas_df.columns:
    tactics_col = 'tactics'
elif 'tactics_used' in atlas_df.columns:
    tactics_col = 'tactics_used'
else:
    # Fallback search
    for col in atlas_df.columns:
        if 'tactics' in str(col).lower() and 'n_' not in str(col).lower() and 'question' not in str(col).lower():
            tactics_col = col
            break

if not tactics_col:
    print("Error: Could not identify 'tactics' text column. Available columns with 'tactics':")
    print([c for c in atlas_df.columns if 'tactics' in str(c).lower()])
    sys.exit(1)

print(f"Using column '{tactics_col}' for tactics analysis.")

# Inspect data to ensure it contains text
print("Sample values from tactics column:")
print(atlas_df[tactics_col].dropna().head(5).values)

# Normalize tactics column
atlas_df[tactics_col] = atlas_df[tactics_col].fillna('').astype(str).str.lower()

# 4. Create Binary Flags
# 'Collection' (TA0009) and 'Exfiltration' (TA0010)
def has_tactic(text, names, ids):
    text = text.lower()
    for name in names:
        if name.lower() in text:
            return True
    for tid in ids:
        if tid.lower() in text:
            return True
    return False

atlas_df['has_collection'] = atlas_df[tactics_col].apply(
    lambda x: has_tactic(x, ['collection'], ['ta0009'])
)

atlas_df['has_exfiltration'] = atlas_df[tactics_col].apply(
    lambda x: has_tactic(x, ['exfiltration'], ['ta0010'])
)

# 5. Contingency Table
contingency_table = pd.crosstab(
    atlas_df['has_collection'], 
    atlas_df['has_exfiltration'], 
    rownames=['Has Collection (TA0009)'], 
    colnames=['Has Exfiltration (TA0010)']
)

# Ensure 2x2
contingency_table = contingency_table.reindex(
    index=[False, True], columns=[False, True], fill_value=0
)

print("\nContingency Table:")
print(contingency_table)

# 6. Statistical Test (Fisher's Exact Test)
odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')

print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# 7. Jaccard Similarity
n_collection = atlas_df['has_collection'].sum()
n_exfiltration = atlas_df['has_exfiltration'].sum()
n_both = contingency_table.loc[True, True]
n_union = n_collection + n_exfiltration - n_both

jaccard = n_both / n_union if n_union > 0 else 0.0
print(f"\nJaccard Similarity Coefficient: {jaccard:.4f}")

# 8. Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Adversarial Chains: Collection (TA0009) vs Exfiltration (TA0010)')
plt.tight_layout()
plt.show()