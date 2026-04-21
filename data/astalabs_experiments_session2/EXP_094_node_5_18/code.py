import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import os
import sys

# Robust file loading
filename = 'astalabs_discovery_all_data.csv'
file_path = None
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(os.path.join('..', filename)):
    file_path = os.path.join('..', filename)

if file_path is None:
    print(f"Error: {filename} not found.")
    sys.exit(1)

df = pd.read_csv(file_path, low_memory=False)
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)}")

# --- CLASSIFICATION LOGIC ---

# 1. Sector Classification
sector_col = next((c for c in aiid.columns if 'Sector of Deployment' in str(c)), 'Sector of Deployment')

def classify_sector(row):
    val = str(row.get(sector_col, '')).lower()
    # Based on previous debug output 'human health and social work activities'
    if 'health' in val or 'medic' in val or 'hosp' in val or 'patient' in val:
        return 'Healthcare'
    if 'financ' in val or 'bank' in val or 'insurance' in val or 'trading' in val:
        return 'Financial'
    return None

# 2. Harm Classification
# Correctly using 'description' and 'title' based on previous debug findings
def classify_harm(row):
    # Combine relevant text columns. 'title' might exist, 'description' definitely does.
    text_content = []
    for col in ['title', 'description', 'Alleged harmed or nearly harmed parties']:
        if col in row.index and pd.notna(row[col]):
            text_content.append(str(row[col]))
            
    full_text = " ".join(text_content).lower()
    
    # Keywords
    phys_keys = ['death', 'dead', 'kill', 'injury', 'injured', 'hurt', 'bodily', 'safety', 
                 'accident', 'crash', 'physical harm', 'violence', 'assault', 'patient', 'medical condition']
                 
    econ_keys = ['financial', 'economic', 'money', 'dollar', 'loss', 'fraud', 'scam', 'theft', 
                 'bank', 'credit', 'market', 'price', 'trading', 'loan', 'funds']
    
    has_phys = any(k in full_text for k in phys_keys)
    has_econ = any(k in full_text for k in econ_keys)
    
    if has_phys and not has_econ:
        return 'Physical'
    if has_econ and not has_phys:
        return 'Economic'
    if has_phys and has_econ:
        # If both, we treat as ambiguous for this specific A/B test to ensure clean signals.
        return 'Both'
    return 'Other'

# Apply
aiid['target_sector'] = aiid.apply(classify_sector, axis=1)
aiid['target_harm'] = aiid.apply(classify_harm, axis=1)

# Filter
subset = aiid[
    (aiid['target_sector'].isin(['Healthcare', 'Financial'])) & 
    (aiid['target_harm'].isin(['Physical', 'Economic']))
].copy()

print(f"\nClassified Subset (Exclusive Categories): {len(subset)}")
print(subset.groupby(['target_sector', 'target_harm']).size())

# --- STATISTICAL TEST ---
contingency = pd.crosstab(subset['target_sector'], subset['target_harm'])
print("\n--- Contingency Table ---")
print(contingency)

if contingency.size == 4:
    odds, p = fisher_exact(contingency)
    print(f"\nFisher's Exact Test p-value: {p:.4e}")
    print(f"Odds Ratio: {odds:.4f}")
    
    # Percentages
    pcts = contingency.div(contingency.sum(axis=1), axis=0) * 100
    print("\n--- Row Percentages ---")
    print(pcts)
    
    if p < 0.05:
        print("\nResult: Statistically significant dependence found.")
        h_p = pcts.loc['Healthcare', 'Physical']
        f_e = pcts.loc['Financial', 'Economic']
        print(f"Healthcare -> Physical: {h_p:.1f}%")
        print(f"Financial -> Economic: {f_e:.1f}%")
        if h_p > 50 and f_e > 50:
             print("Hypothesis Supported.")
    else:
        print("\nResult: No statistically significant dependence found.")
else:
    print("\nInsufficient data for test.")
