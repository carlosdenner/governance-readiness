import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import re

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Normalize columns
aiid.columns = [c.strip().lower().replace(' ', '_').replace(':', '').replace('.', '') for c in aiid.columns]

# Identify columns
intent_col = next((c for c in aiid.columns if 'intentional_harm' in c), None)

# Find text columns for keyword search (Title, Description, Summary, Reports)
# Common names in AIID: title, description, summary, reports
potential_text_cols = ['title', 'description', 'summary', 'reports', 'incident_description', 'short_description']
text_cols = [c for c in aiid.columns if any(x in c for x in potential_text_cols)]

print(f"Identified Intent Column: {intent_col}")
print(f"Identified Text Columns: {text_cols}")

# --- 1. Map Intent ---
def map_intent(val):
    if pd.isna(val): return None
    s = str(val).lower()
    if 'yes' in s and 'intentionally' in s:
        return 'Intentional'
    if 'no' in s and 'not intentionally' in s:
        return 'Unintentional'
    return None

aiid['intent_mapped'] = aiid[intent_col].apply(map_intent) if intent_col else None

# --- 2. Map Harm (Keyword Search) ---
# Keywords
financial_keywords = [
    'financial', 'money', 'bank', 'fraud', 'theft', 'scam', 'monetary', 'economic', 
    'credit', 'loan', 'cost', 'fund', 'wallet', 'crypto', 'currency', 'privacy', 
    'surveillance', 'leak', 'data breach', 'identity', 'spy', 'monitor', 'record'
]

physical_keywords = [
    'physical', 'safety', 'death', 'dead', 'kill', 'injury', 'injure', 'hurt', 'harm',
    'accident', 'crash', 'collision', 'hit', 'run over', 'medical', 'patient', 'hospital',
    'health', 'burn', 'explode', 'fire', 'attack', 'assault', 'robot', 'drone', 'autonomous'
]

def classify_harm(row):
    # Aggregate text from all available text columns
    text_content = " "
    for col in text_cols:
        val = row[col]
        if pd.notna(val):
            text_content += str(val) + " "
    
    text_lower = text_content.lower()
    
    has_financial = any(k in text_lower for k in financial_keywords)
    has_physical = any(k in text_lower for k in physical_keywords)
    
    if has_financial and not has_physical:
        return 'Financial/Privacy'
    elif has_physical and not has_financial:
        return 'Physical Safety'
    elif has_financial and has_physical:
        # Conflict resolution: Check for strong physical indicators (death/injury) vs generic 'safety'
        strong_physical = any(k in text_lower for k in ['death', 'dead', 'kill', 'injury', 'crash'])
        if strong_physical:
            return 'Physical Safety'
        else:
            return 'Ambiguous/Mixed'
    else:
        return 'Other'

aiid['harm_derived'] = aiid.apply(classify_harm, axis=1)

# --- 3. Analysis ---
# Filter for mapped intent and mapped harm (excluding Other/Ambiguous)
analysis_df = aiid.dropna(subset=['intent_mapped', 'harm_derived'])
analysis_df = analysis_df[analysis_df['harm_derived'].isin(['Financial/Privacy', 'Physical Safety'])]

print(f"\nTotal Mapped Rows: {len(analysis_df)}")
if len(analysis_df) > 0:
    print("Intent Breakdown:\n", analysis_df['intent_mapped'].value_counts())
    print("Harm Breakdown:\n", analysis_df['harm_derived'].value_counts())
    
    # Contingency Table
    contingency = pd.crosstab(analysis_df['intent_mapped'], analysis_df['harm_derived'])
    print("\nContingency Table:\n", contingency)
    
    # Check sample size for test selection
    total_obs = contingency.to_numpy().sum()
    min_expected = 0
    if contingency.shape == (2,2):
        chi2, p, dof, ex = chi2_contingency(contingency)
        min_expected = ex.min()
    
    if contingency.shape == (2,2):
        if min_expected < 5:
            print("\nSmall sample size detected. Using Fisher's Exact Test.")
            odds_ratio, p_val = fisher_exact(contingency)
            print(f"Fisher's Exact P-value: {p_val:.4e}")
            print(f"Odds Ratio: {odds_ratio:.4f}")
        else:
            print("\nUsing Chi-Square Test.")
            print(f"Chi2 Statistic: {chi2:.4f}")
            print(f"P-value: {p:.4e}")
            
        # Interpretation
        if p_val < 0.05 if 'p_val' in locals() else p < 0.05:
            print("Result: Statistically Significant Association.")
            # Calculate Row Percentages to see direction
            row_pcts = contingency.div(contingency.sum(axis=1), axis=0) * 100
            print("\nRow Percentages:\n", row_pcts)
        else:
            print("Result: No Statistically Significant Association.")
    else:
        print("Contingency table is not 2x2. Cannot perform binary association test.")
else:
    print("No valid data points found after filtering.")
