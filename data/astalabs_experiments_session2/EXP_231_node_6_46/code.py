import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
import sys
import os

# [debug] 
# print("# [debug] Starting mapping and analysis script...")

# Load dataset
filenames = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
file_path = None
for fn in filenames:
    if os.path.exists(fn):
        file_path = fn
        break

if file_path is None:
    print("Error: Dataset file not found.")
    sys.exit(1)

df = pd.read_csv(file_path, low_memory=False)

# 1. Build a Mapping Dictionary for Tactic IDs to Names
# We look for rows where tactic_id and tactic_name are present
map_df = df.dropna(subset=['tactic_id', 'tactic_name'])[['tactic_id', 'tactic_name']].drop_duplicates()
id_to_name = dict(zip(map_df['tactic_id'], map_df['tactic_name']))

# Fallback/Augment: manual typical ATLAS mappings if not found in data, 
# but we prioritize data. 'Evasion' usually maps to 'Defense Evasion' (AML.TA0006)
# 'Poisoning' is tricky as it's often a technique, but let's see if the data defines it as a tactic or if we need to search techniques.

# Check if we have 'Poisoning' in our map
poisoning_ids = [k for k, v in id_to_name.items() if 'poisoning' in str(v).lower()]
evasion_ids = [k for k, v in id_to_name.items() if 'evasion' in str(v).lower()]

print("Found Tactic Mappings:")
print(f"  Poisoning IDs: {poisoning_ids}")
print(f"  Evasion IDs: {evasion_ids}")

# If we didn't find 'Poisoning' in tactics, we might need to look at techniques.
# Let's inspect 'technique_id' and 'technique_name' if they exist, or just search for the strings in 'techniques' column of atlas_cases
# The dataframe has 'step3_incident_coding' which has 'tactics_used' (IDs) and 'techniques_used' (IDs).

# Let's try to find technique mappings as well
tech_map = {}
if 'technique_id' in df.columns and 'technique_name' in df.columns:
    t_map_df = df.dropna(subset=['technique_id', 'technique_name'])[['technique_id', 'technique_name']].drop_duplicates()
    tech_map = dict(zip(t_map_df['technique_id'], t_map_df['technique_name']))

poisoning_tech_ids = [k for k, v in tech_map.items() if 'poisoning' in str(v).lower()]
evasion_tech_ids = [k for k, v in tech_map.items() if 'evasion' in str(v).lower()]

# 2. Prepare Analysis Data
target_table = 'step3_incident_coding'
df_coding = df[df['source_table'] == target_table].copy()

# Gap Counting Function
def count_gaps(gap_str):
    if pd.isna(gap_str) or gap_str == '':
        return 0
    cleaned = str(gap_str).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    if not cleaned.strip():
        return 0
    return len([x.strip() for x in cleaned.split(',') if x.strip()])

df_coding['gap_count'] = df_coding['competency_gaps'].apply(count_gaps)

# Categorization Function
def categorize_case(row):
    # Get lists of IDs
    tactics = str(row.get('tactics_used', '')).split(';')
    techniques = str(row.get('techniques_used', '')).split(';')
    
    tactics = [t.strip() for t in tactics]
    techniques = [t.strip() for t in techniques]
    
    is_evasion = False
    is_poisoning = False
    
    # Check Tactics (using the map we built)
    for tid in tactics:
        name = id_to_name.get(tid, '').lower()
        if 'evasion' in name: is_evasion = True
        if 'poisoning' in name: is_poisoning = True
        # Also check hardcoded known IDs just in case map is incomplete
        if tid == 'AML.TA0006': is_evasion = True # Defense Evasion

    # Check Techniques (using the map)
    for teid in techniques:
        name = tech_map.get(teid, '').lower()
        if 'evasion' in name: is_evasion = True
        if 'poisoning' in name: is_poisoning = True
        # Check known technique IDs if map failed
        if teid == 'AML.T0015': is_evasion = True # Evasion
        if teid == 'AML.T0020': is_poisoning = True # Data Poisoning
        if teid == 'AML.T0021': is_poisoning = True # Model Poisoning

    if is_evasion and not is_poisoning:
        return 'Evasion'
    elif is_poisoning and not is_evasion:
        return 'Poisoning'
    elif is_evasion and is_poisoning:
        return 'Mixed'
    else:
        return 'Other'

df_coding['category'] = df_coding.apply(categorize_case, axis=1)

print("\n--- Analysis Categories ---")
print(df_coding['category'].value_counts())

# 3. Statistical Analysis
evasion_scores = df_coding[df_coding['category'] == 'Evasion']['gap_count']
poisoning_scores = df_coding[df_coding['category'] == 'Poisoning']['gap_count']

print(f"\nEvasion (n={len(evasion_scores)}): Mean={evasion_scores.mean():.2f}")
print(f"Poisoning (n={len(poisoning_scores)}): Mean={poisoning_scores.mean():.2f}")

if len(evasion_scores) > 1 and len(poisoning_scores) > 1:
    t_stat, p_val = ttest_ind(evasion_scores, poisoning_scores, equal_var=False)
    print(f"\nWelch's T-Test: t={t_stat:.4f}, p-value={p_val:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.boxplot([evasion_scores, poisoning_scores], tick_labels=['Evasion', 'Poisoning'])
    plt.title('Competency Gaps: Evasion vs. Poisoning')
    plt.ylabel('Number of Missing Controls')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
else:
    print("\nInsufficient data for statistical comparison.")
