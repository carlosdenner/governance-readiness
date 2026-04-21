import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for ATLAS cases
atlas = df[df['source_table'] == 'atlas_cases'].copy()

print(f"Loaded ATLAS cases: {len(atlas)} rows")
# Identify correct column names for tactics and type
# Based on previous failure, we suspect columns are 'tactics' and 'type' not '92_tactics' etc.
tactics_col = 'tactics' if 'tactics' in atlas.columns else '92_tactics'
type_col = 'type' if 'type' in atlas.columns else '91_type'

print(f"Using columns: Tactics='{tactics_col}', Type='{type_col}'")

# Helper to count unique tactics
def count_tactics(row):
    tactic_str = row.get(tactics_col, '')
    if pd.isna(tactic_str) or str(tactic_str).strip() == '':
        return 0
    # Normalize separators (semicolon or comma)
    t_str = str(tactic_str).replace(',', ';')
    # Split, strip whitespace, and count unique non-empty entries
    tactics = [t.strip() for t in t_str.split(';') if t.strip()]
    return len(set(tactics))

# Calculate tactic counts
atlas['tactic_count'] = atlas.apply(count_tactics, axis=1)

# Categorize Attack Type
def categorize_attack(row):
    # Combine type, tactics, name, and summary for robust keyword search
    text_content = (
        str(row.get(type_col, '')) + ' ' + 
        str(row.get(tactics_col, '')) + ' ' + 
        str(row.get('name', '')) + ' ' + 
        str(row.get('summary', ''))
    ).lower()
    
    # Keywords for Exfiltration (Model Stealing/Inversion)
    exfil_keywords = ['exfiltration', 'model stealing', 'model inversion', 'extraction', 'steal']
    
    # Keywords for Evasion (Adversarial Example)
    evasion_keywords = ['evasion', 'adversarial example', 'perturbation', 'noise']
    
    # Classification logic
    if any(k in text_content for k in exfil_keywords):
        return 'Exfiltration'
    elif any(k in text_content for k in evasion_keywords):
        return 'Evasion'
    else:
        return 'Other'

atlas['attack_category'] = atlas.apply(categorize_attack, axis=1)

# Filter for the two groups
cohorts = atlas[atlas['attack_category'].isin(['Exfiltration', 'Evasion'])].copy()

# Stats
print("\n--- Cohort Analysis: Kill-Chain Complexity ---")
print(cohorts['attack_category'].value_counts())
group_stats = cohorts.groupby('attack_category')['tactic_count'].describe()
print(group_stats)

exfil_counts = cohorts[cohorts['attack_category'] == 'Exfiltration']['tactic_count']
evasion_counts = cohorts[cohorts['attack_category'] == 'Evasion']['tactic_count']

# T-Test
if len(exfil_counts) > 1 and len(evasion_counts) > 1:
    t_stat, p_val = stats.ttest_ind(exfil_counts, evasion_counts, equal_var=False)
    print(f"\nIndependent T-Test Results:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Result: Statistically significant difference detected (p < 0.05).")
    else:
        print("Result: No statistically significant difference detected.")
else:
    print("\nInsufficient data for T-test.")

# Visualization
plt.figure(figsize=(10, 6))
# Using simple boxplot if violin fails or data is sparse, but violin is requested.
sns.violinplot(x='attack_category', y='tactic_count', data=cohorts, palette='muted', inner='stick')
plt.title('Adversarial Kill-Chain Complexity: Exfiltration vs Evasion')
plt.xlabel('Attack Category')
plt.ylabel('Number of Unique Tactics Used')
plt.show()
