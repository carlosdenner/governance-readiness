import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

# Load the dataset
ds_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(ds_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../' + ds_path, low_memory=False)

# Filter for relevant tables
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
coding_df = df[df['source_table'] == 'step3_incident_coding'].copy()

# Merge on 'name'
merged_df = pd.merge(atlas_df[['name', 'tactics']], 
                     coding_df[['name', 'competency_domains']], 
                     on='name', how='inner')

# Refined Classification Logic to avoid "everything is True"
# We will look for specific keywords associated with the Hypothesis mechanism

def classify_row(row):
    tactics = str(row.get('tactics', '')).upper()
    domains = str(row.get('competency_domains', '')).upper()
    
    # Tactics
    is_exfil = 'EXFILTRATION' in tactics
    is_evasion = 'EVASION' in tactics
    
    # Integration Gaps (Focus on Access/Architectural controls as per hypothesis)
    # Keywords: Access Boundary, Privilege, Identity, Network, Supply Chain
    int_keywords = ['ACCESS BOUNDARY', 'PRIVILEGE', 'IDENTITY', 'NETWORK', 'SUPPLY CHAIN']
    has_integration_gap = any(k in domains for k in int_keywords)
    
    # Trust Gaps (Focus on Model Robustness/Evasion Defense as per hypothesis)
    # Keywords: Defense Evasion, Robustness, Model Access, Adversarial
    # Note: 'Defense Evasion' appears in both tactic and gap names, ensure we look at domains column
    trust_keywords = ['DEFENSE EVASION', 'ROBUSTNESS', 'MODEL ACCESS', 'ADVERSARIAL']
    has_trust_gap = any(k in domains for k in trust_keywords)
    
    return pd.Series([is_exfil, is_evasion, has_integration_gap, has_trust_gap])

merged_df[['is_exfil', 'is_evasion', 'has_int_gap', 'has_trust_gap']] = merged_df.apply(classify_row, axis=1)

# --- Analysis 1: Exfiltration vs Integration Gaps ---
# Hypothesis: Exfiltration tactics are associated with Integration Gaps
print("\n--- Analysis 1: Exfiltration vs Integration (Access/Network) Gaps ---")
cont_exfil = pd.crosstab(merged_df['is_exfil'], merged_df['has_int_gap'], 
                         rownames=['Tactic: Exfiltration'], colnames=['Gap: Integration'])
print(cont_exfil)

if cont_exfil.size == 4:
    odds_exfil, p_exfil = fisher_exact(cont_exfil)
    print(f"Fisher Exact p-value: {p_exfil:.4f}")
    print(f"Odds Ratio: {odds_exfil:.4f}")
else:
    print("Degenerate table")

# --- Analysis 2: Evasion vs Trust Gaps ---
# Hypothesis: Evasion tactics are associated with Trust (Robustness) Gaps
print("\n--- Analysis 2: Evasion vs Trust (Robustness/Model) Gaps ---")
cont_evasion = pd.crosstab(merged_df['is_evasion'], merged_df['has_trust_gap'], 
                           rownames=['Tactic: Evasion'], colnames=['Gap: Trust'])
print(cont_evasion)

if cont_evasion.size == 4:
    odds_evasion, p_evasion = fisher_exact(cont_evasion)
    print(f"Fisher Exact p-value: {p_evasion:.4f}")
    print(f"Odds Ratio: {odds_evasion:.4f}")
else:
    print("Degenerate table")

# Correlation Matrix for visibility
print("\n--- Correlation Matrix (Phi Coefficient approximation) ---")
corr_matrix = merged_df[['is_exfil', 'is_evasion', 'has_int_gap', 'has_trust_gap']].corr()
print(corr_matrix)
