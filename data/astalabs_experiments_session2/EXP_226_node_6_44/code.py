import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for AIID Incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)}")

# 3. Create Text Blob for Classification
aiid_df['text_blob'] = (aiid_df['title'].fillna('') + ' ' + aiid_df['description'].fillna('')).str.lower()

# 4. Define Keyword Lists
# Sector Keywords
health_keywords = ['health', 'medic', 'doctor', 'patient', 'hosp', 'surg', 'clinic', 'cancer', 'radiolog', 'triage', 'diagnos', 'disease', 'treatment']
trans_keywords = ['transport', 'vehicle', 'car', 'driv', 'autonomous', 'truck', 'bus', 'tesla', 'uber', 'crash', 'autopilot', 'traffic', 'aviation', 'plane', 'accident', 'road', 'highway']

# Harm Keywords
equity_keywords = ['bias', 'discriminat', 'racist', 'sexist', 'fairness', 'gender', 'race', 'demographic', 'allocati', 'credit', 'hiring', 'loan', 'profile', 'stereotype', 'minority', 'women', 'black', 'white', 'asian', 'latino']
safety_keywords = ['injur', 'kill', 'death', 'accident', 'collision', 'crash', 'hurt', 'wound', 'physical', 'safety', 'fatal', 'perform', 'error', 'fail', 'malfunction', 'stuck', 'hit', 'damage']

# 5. Classification Functions
def classify_sector(text):
    is_health = any(k in text for k in health_keywords)
    is_trans = any(k in text for k in trans_keywords)
    
    if is_health and not is_trans:
        return 'Healthcare'
    elif is_trans and not is_health:
        return 'Transportation'
    elif is_health and is_trans:
        # Conflict resolution: count occurrences
        h_count = sum(text.count(k) for k in health_keywords)
        t_count = sum(text.count(k) for k in trans_keywords)
        return 'Healthcare' if h_count > t_count else 'Transportation'
    return 'Other'

def classify_harm(text):
    is_equity = any(k in text for k in equity_keywords)
    is_safety = any(k in text for k in safety_keywords)
    
    if is_equity and not is_safety:
        return 'Equity/Allocative'
    elif is_safety and not is_equity:
        return 'Safety/Performance'
    elif is_equity and is_safety:
        # Conflict resolution
        e_count = sum(text.count(k) for k in equity_keywords)
        s_count = sum(text.count(k) for k in safety_keywords)
        return 'Equity/Allocative' if e_count > s_count else 'Safety/Performance'
    return 'Other'

# 6. Apply Classification
aiid_df['predicted_sector'] = aiid_df['text_blob'].apply(classify_sector)
aiid_df['predicted_harm'] = aiid_df['text_blob'].apply(classify_harm)

# 7. Filter for Target Groups
target_df = aiid_df[
    (aiid_df['predicted_sector'].isin(['Healthcare', 'Transportation'])) & 
    (aiid_df['predicted_harm'].isin(['Equity/Allocative', 'Safety/Performance']))
].copy()

print(f"Records after keyword classification and filtering: {len(target_df)}")
print("\nDistribution by Sector:")
print(target_df['predicted_sector'].value_counts())
print("\nDistribution by Harm:")
print(target_df['predicted_harm'].value_counts())

# 8. Contingency Table & Stats
contingency_table = pd.crosstab(target_df['predicted_sector'], target_df['predicted_harm'])
print("\n--- Contingency Table ---")
print(contingency_table)

if not contingency_table.empty and contingency_table.size == 4:
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Calculate Odds Ratio (AD / BC)
    # Table layout: 
    #                 Equity   Safety
    # Healthcare        A        B
    # Transportation    C        D
    
    try:
        A = contingency_table.loc['Healthcare', 'Equity/Allocative']
        B = contingency_table.loc['Healthcare', 'Safety/Performance']
        C = contingency_table.loc['Transportation', 'Equity/Allocative']
        D = contingency_table.loc['Transportation', 'Safety/Performance']
        
        odds_ratio = (A * D) / (B * C) if (B * C) > 0 else np.inf
        print(f"Odds Ratio (Healthcare Equity / Transport Equity): {odds_ratio:.4f}")
        
        h_eq_prop = A / (A + B)
        t_eq_prop = C / (C + D)
        print(f"Healthcare Equity Proportion: {h_eq_prop:.2%}")
        print(f"Transportation Equity Proportion: {t_eq_prop:.2%}")
        
        if p < 0.05:
            print("\nCONCLUSION: Significant difference found.")
            if h_eq_prop > t_eq_prop:
                print("Evidence SUPPORTS hypothesis: Healthcare has higher equity harm rates.")
            else:
                print("Evidence CONTRADICTS hypothesis.")
        else:
             print("\nCONCLUSION: No significant difference found.")
             
        # Plot
        contingency_table.plot(kind='bar', stacked=True)
        plt.title('Harm Distribution: Healthcare vs Transportation (Keyword Classified)')
        plt.ylabel('Incident Count')
        plt.tight_layout()
        plt.show()
        
    except KeyError as e:
        print(f"Error accessing table keys: {e}")
else:
    print("Insufficient data for statistical test.")