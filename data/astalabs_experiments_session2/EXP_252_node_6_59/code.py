import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID incidents loaded: {len(aiid)}")

# Combine title and description for text mining
aiid['text_content'] = aiid['title'].fillna('') + " " + aiid['description'].fillna('')
aiid['text_content'] = aiid['text_content'].str.lower()

# ---------------------------------------------------------
# Text Mining Functions
# ---------------------------------------------------------

def get_sector(text):
    health_kw = ['health', 'hospital', 'medic', 'patient', 'doctor', 'nurse', 'diagnos', 'clinic', 'cancer', 'surgery']
    finance_kw = ['bank', 'financ', 'loan', 'credit', 'trading', 'stock', 'market', 'money', 'invest', 'crypto', 'currency']
    
    is_health = any(k in text for k in health_kw)
    is_finance = any(k in text for k in finance_kw)
    
    if is_health and not is_finance:
        return 'Healthcare'
    if is_finance and not is_health:
        return 'Financial'
    return None  # Ambiguous or neither

def get_harm(text):
    # Physical/Safety keywords
    phys_kw = ['death', 'kill', 'inju', 'physic', 'safety', 'accident', 'crash', 'violen', 'attack', 'collision', 'burn', 'murder', 'died']
    # Economic/Reputational keywords (avoiding 'credit' here if it's too overlapping, but 'fraud' is good)
    econ_kw = ['fraud', 'scam', 'theft', 'monetary', 'bankrupt', 'loss', 'fine', 'penalty', 'reputation', 'steal', 'stolen', 'cost']
    
    is_phys = any(k in text for k in phys_kw)
    is_econ = any(k in text for k in econ_kw)
    
    if is_phys and not is_econ:
        return 'Physical/Safety'
    if is_econ and not is_phys:
        return 'Economic/Reputational'
    # If both, prioritize Physical/Safety as it is the more severe category often distinguishing these sectors
    if is_phys and is_econ:
        return 'Physical/Safety'
    return None

# Apply mappings
aiid['Derived_Sector'] = aiid['text_content'].apply(get_sector)
aiid['Derived_Harm'] = aiid['text_content'].apply(get_harm)

# Filter for valid rows
analysis_df = aiid[aiid['Derived_Sector'].notna() & aiid['Derived_Harm'].notna()].copy()

print(f"\nDerived Data Subset Size: {len(analysis_df)}")
print("Counts by Sector:\n", analysis_df['Derived_Sector'].value_counts())
print("Counts by Harm:\n", analysis_df['Derived_Harm'].value_counts())

# ---------------------------------------------------------
# Statistical Test
# ---------------------------------------------------------
if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
else:
    contingency = pd.crosstab(analysis_df['Derived_Sector'], analysis_df['Derived_Harm'])
    print("\nContingency Table:")
    print(contingency)
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    # Calculate Row Percentages for clarity
    row_probs = contingency.div(contingency.sum(axis=1), axis=0)
    print("\nRow Percentages:")
    print(row_probs)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    # Using a heatmap of counts
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('Heatmap: Sector vs. Inferred Harm Type')
    plt.xlabel('Harm Type')
    plt.ylabel('Sector')
    plt.tight_layout()
    plt.show()
