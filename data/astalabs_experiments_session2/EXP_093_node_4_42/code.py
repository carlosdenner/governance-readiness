import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import os

# Load dataset with fallback for path location
filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    if os.path.exists('../' + filename):
        filename = '../' + filename
    else:
        print(f"Error: {filename} not found in current or parent directory.")

df = pd.read_csv(filename, low_memory=False)

# Filter for ATLAS cases
atlas = df[df['source_table'] == 'atlas_cases'].copy()
print(f"ATLAS cases loaded: {len(atlas)}")

# Normalize tactics column
# Ensure strings, lowercase
atlas['tactics_norm'] = atlas['tactics'].fillna('').astype(str).str.lower()

# Helper to check for tactic presence
def has_tactic(text, tactic_name):
    # Simple substring check, robust to formatting like 'Initial Access' vs 'initial-access'
    return tactic_name in text

# Create binary flags for the tactics of interest
atlas['init_access'] = atlas['tactics_norm'].apply(lambda x: has_tactic(x, 'initial access') or has_tactic(x, 'initial-access'))
atlas['collection'] = atlas['tactics_norm'].apply(lambda x: has_tactic(x, 'collection'))
atlas['exfiltration'] = atlas['tactics_norm'].apply(lambda x: has_tactic(x, 'exfiltration'))

# Define the two groups based on the antecedent tactic
# Group 1: Cases having 'Collection'
# Group 2: Cases having 'Initial Access'

# Get the counts for the denominators (n)
n_coll = atlas['collection'].sum()
n_init = atlas['init_access'].sum()

# Get the counts for the numerators (k) - i.e., having Exfiltration given the group
k_coll_exfil = atlas[atlas['collection']]['exfiltration'].sum()
k_init_exfil = atlas[atlas['init_access']]['exfiltration'].sum()

# Calculate Probabilities
p_exfil_given_coll = k_coll_exfil / n_coll if n_coll > 0 else 0
p_exfil_given_init = k_init_exfil / n_init if n_init > 0 else 0

print("\n--- Co-occurrence Statistics ---")
print(f"P(Exfiltration | Collection)     = {k_coll_exfil}/{n_coll} ({p_exfil_given_coll:.2%})")
print(f"P(Exfiltration | Initial Access) = {k_init_exfil}/{n_init} ({p_exfil_given_init:.2%})")

# Perform Z-test for difference of proportions
# Note: This assumes independent samples, which is a limitation here as cases can be in both groups.
if n_coll > 0 and n_init > 0:
    count = np.array([k_coll_exfil, k_init_exfil])
    nobs = np.array([n_coll, n_init])
    
    stat, pval = proportions_ztest(count, nobs)
    
    print("\n--- Z-Test Results ---")
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value:     {pval:.4f}")
    
    if pval < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: No statistically significant difference.")
else:
    print("\nInsufficient data for Z-test.")

# Visualization
plt.figure(figsize=(8, 5))
probs = [p_exfil_given_coll, p_exfil_given_init]
labels = ['Given Collection', 'Given Initial Access']
colors = ['#1f77b4', '#ff7f0e']

bars = plt.bar(labels, probs, color=colors, alpha=0.8)
plt.ylabel('Probability of Exfiltration')
plt.title('Conditional Probability of Exfiltration Tactic')
plt.ylim(0, 1.0)

for bar, p, k, n in zip(bars, probs, [k_coll_exfil, k_init_exfil], [n_coll, n_init]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f"{p:.1%} (n={k}/{n})", ha='center', va='bottom', fontsize=10)

plt.show()