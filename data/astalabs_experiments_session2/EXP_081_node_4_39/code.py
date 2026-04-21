import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

# 1. Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df_all = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df_all = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for 'aiid_incidents' and process dates
df_incidents = df_all[df_all['source_table'] == 'aiid_incidents'].copy()
df_incidents['date_parsed'] = pd.to_datetime(df_incidents['date'], errors='coerce')
df_incidents = df_incidents.dropna(subset=['date_parsed'])
df_incidents['year'] = df_incidents['date_parsed'].dt.year

# 3. Bin into Epochs
df_incidents['epoch'] = df_incidents['year'].apply(lambda x: 'Pre-2020' if x < 2020 else 'Post-2020')

# 4. Infer Harm Domains based on keywords in Title and Description
# Keywords definitions
keywords = {
    'Physical': ['death', 'dead', 'kill', 'injur', 'physical', 'health', 'safe', 'accident', 'crash', 'collision', 'bodily', 'harm'],
    'Economic': ['economic', 'financ', 'money', 'dollar', 'cost', 'job', 'employ', 'fraud', 'scam', 'market', 'theft', 'bank'],
    'Societal': ['societ', 'bias', 'discriminat', 'raci', 'sexis', 'gender', 'politic', 'elect', 'democra', 'surveil', 'priva', 'civil right', 'unfair', 'inequity', 'stereo'],
    'Psychological': ['psycholog', 'mental', 'emotion', 'fear', 'terror', 'trauma', 'stress', 'harass', 'manipulat', 'anxiety']
}

def infer_harm(row):
    text = (str(row.get('title', '')) + ' ' + str(row.get('description', ''))).lower()
    detected_harms = []
    for domain, kw_list in keywords.items():
        for kw in kw_list:
            if kw in text:
                detected_harms.append(domain)
                break # Found one keyword for this domain, move to next domain
    return detected_harms

df_incidents['inferred_harms'] = df_incidents.apply(infer_harm, axis=1)

# 5. Explode the list of harms to handle multi-label incidents
df_exploded = df_incidents.explode('inferred_harms')

# Drop rows where no harm was inferred
df_exploded = df_exploded.dropna(subset=['inferred_harms'])

# *** CRITICAL FIX: Reset index to avoid ValueError in crosstab ***
df_exploded = df_exploded.reset_index(drop=True)

# 6. Generate Summary Statistics and Contingency Table
contingency_table = pd.crosstab(df_exploded['epoch'], df_exploded['inferred_harms'])

print("--- Inferred Harm Domain Analysis ---")
print(f"Total incidents analyzed: {len(df_incidents)}")
print(f"Incidents with inferred harms: {df_incidents['inferred_harms'].map(lambda x: len(x) > 0).sum()}")
print("\nContingency Table (Epoch vs Inferred Harm):")
print(contingency_table)

# Normalize row-wise to see shifts in dominance
contingency_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print("\nProportional Distribution by Epoch:")
print(contingency_norm)

# 7. Statistical Test (Chi-Square)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically significant shift in Harm Domain distribution detected.")
else:
    print("Result: No statistically significant shift detected.")

# 8. Visualization
plt.figure(figsize=(10, 6))
contingency_norm.plot(kind='bar', stacked=False, ax=plt.gca())
plt.title('Shift in Inferred AI Harm Domains: Pre-2020 vs Post-2020')
plt.xlabel('Epoch')
plt.ylabel('Proportion of Mentions within Epoch')
plt.legend(title='Harm Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Validate specific claims
# Claim 1: Pre-2020 'Physical' & 'Economic' dominated
pre_phys = contingency_norm.loc['Pre-2020', 'Physical']
pre_econ = contingency_norm.loc['Pre-2020', 'Economic']
print(f"\nPre-2020 Physical + Economic share: {pre_phys + pre_econ:.2%}")

# Claim 2: Post-2020 'Societal' & 'Psychological' become primary
post_soc = contingency_norm.loc['Post-2020', 'Societal']
post_psy = contingency_norm.loc['Post-2020', 'Psychological']
print(f"Post-2020 Societal + Psychological share: {post_soc + post_psy:.2%}")

# Comparison
pre_soc_psy = contingency_norm.loc['Pre-2020', 'Societal'] + contingency_norm.loc['Pre-2020', 'Psychological']
post_phys_econ = contingency_norm.loc['Post-2020', 'Physical'] + contingency_norm.loc['Post-2020', 'Economic']

print(f"\nShift Analysis:")
print(f"Physical/Economic: {pre_phys + pre_econ:.2%} (Pre) -> {post_phys_econ:.2%} (Post)")
print(f"Societal/Psychological: {pre_soc_psy:.2%} (Pre) -> {post_soc + post_psy:.2%} (Post)")