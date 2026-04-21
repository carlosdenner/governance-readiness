# [debug]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found in ../ or current directory.")
        exit(1)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()

print(f"Loaded ATLAS cases: {len(atlas_df)}")
print("Sample of 'tactics' column:")
print(atlas_df['tactics'].head(10).tolist())
print("Sample of 'techniques' column:")
print(atlas_df['techniques'].head(10).tolist())

# Helper function to check for terms
def check_term(row, column, terms):
    val = str(row[column]).lower()
    for t in terms:
        if t.lower() in val:
            return True
    return False

# Define detection logic
# Exfiltration is a tactic
# Evasion is a tactic (often 'Defense Evasion')
# Poisoning is often a technique, but we check both

atlas_df['has_exfiltration'] = atlas_df.apply(lambda x: check_term(x, 'tactics', ['Exfiltration']), axis=1)
atlas_df['has_evasion'] = atlas_df.apply(lambda x: check_term(x, 'tactics', ['Evasion', 'Defense Evasion']), axis=1)
atlas_df['has_poisoning'] = atlas_df.apply(lambda x: check_term(x, 'tactics', ['Poisoning']) or check_term(x, 'techniques', ['Poisoning']), axis=1)

print("\nCounts:")
print(f"Exfiltration: {atlas_df['has_exfiltration'].sum()}")
print(f"Evasion: {atlas_df['has_evasion'].sum()}")
print(f"Poisoning: {atlas_df['has_poisoning'].sum()}")

# Co-occurrence counts with Exfiltration
exfil_evasion = atlas_df[(atlas_df['has_exfiltration']) & (atlas_df['has_evasion'])].shape[0]
exfil_poisoning = atlas_df[(atlas_df['has_exfiltration']) & (atlas_df['has_poisoning'])].shape[0]
exfil_only = atlas_df[(atlas_df['has_exfiltration']) & (~atlas_df['has_evasion']) & (~atlas_df['has_poisoning'])].shape[0]

# Total Exfiltration cases
total_exfil = atlas_df['has_exfiltration'].sum()

print(f"\nCo-occurrences with Exfiltration (Total Exfil Cases: {total_exfil}):")
print(f"Exfiltration + Evasion: {exfil_evasion}")
print(f"Exfiltration + Poisoning: {exfil_poisoning}")

# Jaccard Similarity
# J(A,B) = |A n B| / |A u B|
def jaccard(col1, col2):
    intersection = (atlas_df[col1] & atlas_df[col2]).sum()
    union = (atlas_df[col1] | atlas_df[col2]).sum()
    return intersection / union if union > 0 else 0

j_exfil_evasion = jaccard('has_exfiltration', 'has_evasion')
j_exfil_poisoning = jaccard('has_exfiltration', 'has_poisoning')

print(f"\nJaccard Similarity (Exfiltration, Evasion): {j_exfil_evasion:.4f}")
print(f"Jaccard Similarity (Exfiltration, Poisoning): {j_exfil_poisoning:.4f}")

# Fisher's Exact Tests
# 1. Association between Exfiltration and Evasion
# [[Exfil & Evasion, Exfil & !Evasion],
#  [!Exfil & Evasion, !Exfil & !Evasion]]
ct_evasion = pd.crosstab(atlas_df['has_exfiltration'], atlas_df['has_evasion'])
_, p_evasion = fisher_exact(ct_evasion)
print(f"\nFisher's Test (Exfiltration <-> Evasion) p-value: {p_evasion:.4f}")
print("Contingency Table (Exfil vs Evasion):")
print(ct_evasion)

# 2. Association between Exfiltration and Poisoning
ct_poisoning = pd.crosstab(atlas_df['has_exfiltration'], atlas_df['has_poisoning'])
_, p_poisoning = fisher_exact(ct_poisoning)
print(f"\nFisher's Test (Exfiltration <-> Poisoning) p-value: {p_poisoning:.4f}")
print("Contingency Table (Exfil vs Poisoning):")
print(ct_poisoning)

# Prepare data for plotting
co_occur_data = {
    'Pair': ['Exfiltration-Evasion', 'Exfiltration-Poisoning'],
    'Count': [exfil_evasion, exfil_poisoning],
    'Jaccard': [j_exfil_evasion, j_exfil_poisoning]
}

plt.figure(figsize=(10, 5))

# Plot Co-occurrence Counts
plt.subplot(1, 2, 1)
sns.barplot(x='Pair', y='Count', data=co_occur_data)
plt.title('Co-occurrence Counts')
plt.ylabel('Number of Cases')

# Plot Jaccard Similarity
plt.subplot(1, 2, 2)
sns.barplot(x='Pair', y='Jaccard', data=co_occur_data)
plt.title('Jaccard Similarity')
plt.ylabel('Similarity Index')

plt.tight_layout()
plt.show()