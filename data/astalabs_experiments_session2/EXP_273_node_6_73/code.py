import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(aiid)} AIID incidents.")

# Dynamic column finding
cols = aiid.columns.tolist()
intent_col = next((c for c in cols if 'Intentional Harm' in str(c)), 'Intentional Harm')
harm_level_col = next((c for c in cols if 'AI Harm Level' in str(c)), 'AI Harm Level')
tangible_harm_col = next((c for c in cols if 'Tangible Harm' in str(c)), 'Tangible Harm')

print(f"Columns identified:\n - Intent: {intent_col}\n - Harm Level: {harm_level_col}\n - Tangible Harm: {tangible_harm_col}")

# --- 1. Clean Intentionality ---
def clean_intent(val):
    s = str(val).lower().strip()
    if s.startswith('yes'):
        return True
    elif s.startswith('no'):
        return False
    return np.nan

aiid['is_intentional'] = aiid[intent_col].apply(clean_intent)

# --- 2. Construct Harm Severity Score ---
# Strategy: Use 'Tangible Harm' for granularity. If specific keywords found, assign score.
# If not found, look at 'AI Harm Level' as fallback context.

def calculate_severity(row):
    # Get strings
    t_harm = str(row[tangible_harm_col]).lower() if pd.notna(row[tangible_harm_col]) else ''
    h_level = str(row[harm_level_col]).lower() if pd.notna(row[harm_level_col]) else ''
    
    # Priority 1: High Severity Keywords in Tangible Harm description
    if any(x in t_harm for x in ['death', 'killed', 'loss of life', 'fatal']):
        return 4
    if any(x in t_harm for x in ['injury', 'physical', 'hospital', 'safety']):
        return 3
    if any(x in t_harm for x in ['financial', 'economic', 'property', 'monetary', 'theft']):
        return 2
    if any(x in t_harm for x in ['reputation', 'psychological', 'bias', 'discrimination', 'privacy', 'civil rights']):
        return 1
        
    # Priority 2: Fallback to Broad Categories in AI Harm Level
    if 'event' in h_level:
        # Default for an event with unspecified tangible harm
        return 1.5 # Treat as generic harm
    if 'issue' in h_level:
        return 1
    if 'near-miss' in h_level or 'near miss' in h_level:
        return 0
    if 'none' in h_level:
        return 0
        
    return np.nan

aiid['severity_score'] = aiid.apply(calculate_severity, axis=1)

# --- 3. Analysis ---
valid = aiid.dropna(subset=['is_intentional', 'severity_score'])
print(f"\nValid rows for analysis: {len(valid)}")

if len(valid) > 10:
    # Descriptive Stats
    stats = valid.groupby('is_intentional')['severity_score'].agg(['count', 'mean', 'median', 'std'])
    print("\n--- Descriptive Statistics by Intentionality ---")
    print(stats)
    
    # Mann-Whitney U Test
    group_intent = valid[valid['is_intentional'] == True]['severity_score']
    group_unintent = valid[valid['is_intentional'] == False]['severity_score']
    
    stat, p_val = mannwhitneyu(group_intent, group_unintent, alternative='greater') 
    # Hypothesis: Intentional > Unintentional (one-sided 'greater')
    
    print(f"\nMann-Whitney U Test (Intentional > Unintentional):\nStatistic={stat:.2f}, p-value={p_val:.5f}")
    
    if p_val < 0.05:
        print("RESULT: Statistically Significant. Intentional incidents have higher severity.")
    else:
        print("RESULT: Not Statistically Significant.")
        
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='is_intentional', y='severity_score', data=valid, inner='box', palette='Set2')
    plt.xticks([0, 1], ['Unintentional', 'Intentional'])
    plt.ylabel('Harm Severity Score (0=Near Miss -> 4=Fatal)')
    plt.title('Comparison of Harm Severity: Intentional vs Unintentional AI Incidents')
    plt.show()
    
else:
    print("Insufficient data. Printing sample of Tangible Harm for debugging:")
    print(aiid[tangible_harm_col].unique()[:20])
