import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
ds_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(ds_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded AIID incidents: {len(aiid_df)}")

# --- 1. Map Autonomy ---
# Observed values: 'Autonomy1', 'Autonomy2', 'Autonomy3'
# Assumption: 1=Low, 2=Medium, 3=High. Hypothesis compares High vs Low/Medium.
autonomy_col = 'Autonomy Level'
if autonomy_col not in aiid_df.columns:
    # Fallback search
    cols = [c for c in aiid_df.columns if 'autonomy' in c.lower()]
    if cols:
        autonomy_col = cols[0]

print(f"Using Autonomy column: {autonomy_col}")

def map_autonomy_level(val):
    s = str(val).lower()
    if 'autonomy3' in s or 'high' in s or 'full' in s:
        return 'High'
    if 'autonomy1' in s or 'autonomy2' in s or 'low' in s or 'medium' in s:
        return 'Low'
    return np.nan

aiid_df['Autonomy_Bin'] = aiid_df[autonomy_col].apply(map_autonomy_level)
print("Autonomy distribution:\n", aiid_df['Autonomy_Bin'].value_counts())

# --- 2. Map Harm Type ---
# 'Harm Domain' is boolean (yes/no), so we must extract type from text description.

# Identify the best text column
text_candidates = ['description', 'summary', 'title', 'incident_description']
text_col = None

# 1. Try explicit names
for c in text_candidates:
    if c in aiid_df.columns:
        text_col = c
        break

# 2. If not found, try searching column names
if not text_col:
    for c in aiid_df.columns:
        if 'description' in c.lower() or 'summary' in c.lower():
            text_col = c
            break

# 3. If still not found, find the object column with highest average length
if not text_col:
    object_cols = aiid_df.select_dtypes(include=['object']).columns
    best_col = None
    max_len = 0
    for c in object_cols:
        # Sample first 100 non-nulls
        sample = aiid_df[c].dropna().head(100).astype(str)
        if len(sample) > 0:
            avg_len = sample.str.len().mean()
            if avg_len > max_len:
                max_len = avg_len
                best_col = c
    if max_len > 30: # Threshold to ensure it's not just a long ID
        text_col = best_col

print(f"Using Text column for Harm classification: {text_col}")

def map_harm_type(text):
    if pd.isna(text):
        return np.nan
    t = str(text).lower()
    
    # Keywords
    physical = ['death', 'dead', 'kill', 'injur', 'hurt', 'fatal', 'accident', 'crash', 'collision', 'safety', 'physical', 'bodily', 'life', 'medical']
    financial_intangible = ['financ', 'money', 'dollar', 'cost', 'credit', 'bank', 'fraud', 'scam', 'discriminat', 'bias', 'racis', 'sexis', 'reputation', 'privacy', 'surveillance', 'rights', 'civil']
    
    # Priority: Physical (since hypothesis asks if autonomy escalates to physical)
    if any(k in t for k in physical):
        return 'Physical'
    if any(k in t for k in financial_intangible):
        return 'Financial/Intangible'
    
    return 'Other/Unclear'

if text_col:
    aiid_df['Harm_Bin'] = aiid_df[text_col].apply(map_harm_type)
else:
    print("No suitable text column found for harm classification.")
    aiid_df['Harm_Bin'] = np.nan

print("Harm distribution:\n", aiid_df['Harm_Bin'].value_counts())

# --- 3. Analysis ---
df_clean = aiid_df.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])
df_clean = df_clean[df_clean['Harm_Bin'] != 'Other/Unclear']

print(f"Records available for analysis: {len(df_clean)}")

if len(df_clean) > 0:
    ct = pd.crosstab(df_clean['Autonomy_Bin'], df_clean['Harm_Bin'])
    print("\nContingency Table:")
    print(ct)
    
    # Plot
    ct.plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'salmon'])
    plt.title('Harm Type by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Count of Incidents')
    plt.xticks(rotation=0)
    plt.legend(title='Harm Type')
    plt.tight_layout()
    plt.show()
    
    # Stats
    chi2, p, dof, ex = chi2_contingency(ct)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Calculate proportions
    props = pd.crosstab(df_clean['Autonomy_Bin'], df_clean['Harm_Bin'], normalize='index')
    print("\nProportions (Row-wise):")
    print(props)
    
    # Hypothesis Check
    high_phys = props.loc['High', 'Physical'] if 'High' in props.index and 'Physical' in props.columns else 0
    low_phys = props.loc['Low', 'Physical'] if 'Low' in props.index and 'Physical' in props.columns else 0
    
    print(f"\nPhysical Harm Rate - High Autonomy: {high_phys:.1%}")
    print(f"Physical Harm Rate - Low Autonomy: {low_phys:.1%}")
    
    if p < 0.05:
        print("Result: Significant difference detected.")
    else:
        print("Result: No significant difference detected.")
else:
    print("Insufficient data for statistical testing.")
