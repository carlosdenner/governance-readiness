import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import sys
import numpy as np

print("Starting experiment: The Intentionality of Greed (Attempt 5 - Text Analysis)...")

# 1. Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(aiid_df)} AIID incidents.")

# 3. Clean Intentionality
# Using the logic validated in previous steps
intent_col = 'Intentional Harm'
if intent_col not in aiid_df.columns:
    cols = [c for c in aiid_df.columns if 'Intentional Harm' in str(c)]
    if cols: 
        intent_col = cols[0]
    else:
        print("Column 'Intentional Harm' not found.")
        sys.exit(1)

def map_intentionality(val):
    s = str(val).lower()
    if 'yes. intentionally' in s:
        return 1
    elif 'no. not intentionally' in s:
        return 0
    return None

aiid_df['is_intentional'] = aiid_df[intent_col].apply(map_intentionality)
analysis_df = aiid_df.dropna(subset=['is_intentional']).copy()
print(f"Records with valid intentionality: {len(analysis_df)}")

# 4. Infer Harm Domain from Description/Title
# Since structured columns failed, we mine the text.
analysis_df['text_content'] = analysis_df['title'].fillna('') + " " + analysis_df['description'].fillna('')
analysis_df['text_content'] = analysis_df['text_content'].str.lower()

def infer_domain(text):
    # Keywords
    financial_keys = ['financial', 'money', 'cost', 'dollar', 'bank', 'fraud', 'theft', 
                      'market', 'stock', 'economy', 'economic', 'credit', 'price', 'fund', 
                      'wage', 'salary', 'billing', 'fee', 'crypto', 'currency']
    
    physical_keys = ['death', 'dead', 'kill', 'die', 'injury', 'injure', 'hurt', 'physical', 
                     'crash', 'collision', 'accident', 'safety', 'robot', 'autonomous vehicle', 
                     'drone', 'weapon', 'assault', 'violence', 'hospital', 'medical']
    
    civil_keys = ['discrimination', 'discriminat', 'bias', 'racist', 'sexist', 'gender', 'race', 
                  'black', 'white', 'woman', 'man', 'arrest', 'police', 'surveillance', 'privacy', 
                  'facial recognition', 'civil rights', 'liberties', 'censorship', 'profile', 'profiling']
    
    # Check presence
    has_fin = any(k in text for k in financial_keys)
    has_phy = any(k in text for k in physical_keys)
    has_civ = any(k in text for k in civil_keys)
    
    # Priority resolution if multiple match (Physical > Civil > Financial for categorization purposes, 
    # though the hypothesis focuses on Financial. We assign primarily based on what the text *likely* is about.)
    # Let's count matches to be smarter? No, simple priority for now to avoid over-complication.
    
    if has_phy: return 'Physical'
    if has_civ: return 'Civil Rights'
    if has_fin: return 'Financial'
    return 'Other'

analysis_df['inferred_domain'] = analysis_df['text_content'].apply(infer_domain)

# 5. Generate Statistics
summary = analysis_df.groupby('inferred_domain')['is_intentional'].agg(['count', 'mean', 'sum'])
summary.columns = ['Total Incidents', 'Intentionality Rate', 'Intentional Count']
print("\nSummary Statistics by Inferred Harm Domain:")
print(summary)

# 6. Statistical Tests
contingency_table = pd.crosstab(analysis_df['inferred_domain'], analysis_df['is_intentional'])
print("\nContingency Table (0=Unintentional, 1=Intentional):")
print(contingency_table)

# Chi-Square
if len(summary) > 1 and contingency_table.sum().sum() > 0:
    try:
        chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
        print(f"\nOverall Chi-Square Test: Chi2={chi2:.4f}, p-value={p:.4e}")
    except ValueError:
        print("Chi-square failed.")

# Pairwise Comparisons (Financial vs Others)
target = 'Financial'
others = ['Physical', 'Civil Rights']

print(f"\nPairwise Comparisons ({target} vs X):")
for other in others:
    if target in summary.index and other in summary.index:
        subset = analysis_df[analysis_df['inferred_domain'].isin([target, other])]
        ct_sub = pd.crosstab(subset['inferred_domain'], subset['is_intentional'])
        
        # Check if we have both 0s and 1s in the subset to avoid errors
        if ct_sub.shape == (2, 2):
            c2, pv, _, _ = stats.chi2_contingency(ct_sub)
            r1 = summary.loc[target, 'Intentionality Rate']
            r2 = summary.loc[other, 'Intentionality Rate']
            print(f"  {target} ({r1:.1%}) vs {other} ({r2:.1%}): Chi2={c2:.4f}, p={pv:.4e}")
        else:
            print(f"  {target} vs {other}: Insufficient variance (likely 0 intentional in one group).")
    else:
        print(f"  {other} category missing from data.")

# 7. Visualization
if not summary.empty:
    plt.figure(figsize=(10, 6))
    
    # Standard Error
    summary['se'] = summary.apply(lambda row: 
        stats.sem(analysis_df[analysis_df['inferred_domain'] == row.name]['is_intentional']) 
        if row['Total Incidents'] > 1 else 0, axis=1)
    
    sns.barplot(x=summary.index, y=summary['Intentionality Rate'], hue=summary.index, palette='coolwarm', legend=False)
    plt.errorbar(x=range(len(summary)), y=summary['Intentionality Rate'], 
                 yerr=summary['se'], fmt='none', c='black', capsize=5)
    
    plt.title('Intentionality Rate by Inferred Harm Domain')
    plt.ylabel('Proportion of Intentional Incidents')
    plt.xlabel('Harm Domain (Inferred)')
    plt.axhline(y=analysis_df['is_intentional'].mean(), color='gray', linestyle='--', label='Global Average')
    plt.legend()
    plt.tight_layout()
    plt.show()
