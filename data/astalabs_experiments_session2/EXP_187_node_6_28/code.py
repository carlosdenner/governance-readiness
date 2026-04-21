import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# [debug]
print("Starting experiment...")

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)}")

# 3. Sector Classification
# Use 'Sector of Deployment' or find the relevant column
sector_cols = [c for c in aiid_df.columns if 'Sector' in c and 'Deployment' in c]
sector_col = sector_cols[0] if sector_cols else 'Sector of Deployment'

def map_sector(s):
    if pd.isna(s): return None
    s_lower = str(s).lower()
    if 'financ' in s_lower or 'bank' in s_lower or 'insurance' in s_lower:
        return 'Financial'
    elif 'transport' in s_lower or 'automotive' in s_lower or 'aviation' in s_lower:
        return 'Transportation'
    return None

aiid_df['analyzed_sector'] = aiid_df[sector_col].apply(map_sector)
subset = aiid_df.dropna(subset=['analyzed_sector']).copy()

print(f"Subset after sector filtering: {len(subset)}")
print(subset['analyzed_sector'].value_counts())

# 4. Harm Classification based on Description
# We use the 'description' column as 'Harm Domain' was insufficient
desc_col = 'description' if 'description' in subset.columns else 'summary'

def map_harm_from_text(text):
    if pd.isna(text): return 'Other'
    text = str(text).lower()
    
    # Keywords
    physical_keys = ['kill', 'death', 'dead', 'die', 'injur', 'fatal', 'accident', 'crash', 'collision', 
                     'hit', 'struck', 'hurt', 'wound', 'physical', 'safety', 'life']
    economic_keys = ['fraud', 'scam', 'monetary', 'financial', 'money', 'loss', 'credit', 'market', 
                     'stock', 'trade', 'trading', 'bank', 'loan', 'price', 'employment', 'job']
    
    has_physical = any(k in text for k in physical_keys)
    has_economic = any(k in text for k in economic_keys)
    
    # Classification Logic (Priority: Physical > Economic if both, though overlap is rare)
    if has_physical:
        return 'Physical'
    elif has_economic:
        return 'Economic'
    else:
        return 'Other'

subset['harm_category'] = subset[desc_col].apply(map_harm_from_text)

print("\nHarm Category Distribution:")
print(subset['harm_category'].value_counts())

# 5. Generate Contingency Table (2x2 focus: Financial/Transportation vs Economic/Physical)
# We filter out 'Other' for the statistical test to test the specific hypothesis of bias
valid_harms = subset[subset['harm_category'].isin(['Economic', 'Physical'])]

contingency_table = pd.crosstab(valid_harms['analyzed_sector'], valid_harms['harm_category'])

# Ensure columns exist
for col in ['Economic', 'Physical']:
    if col not in contingency_table.columns:
        contingency_table[col] = 0
        
# Reorder
contingency_table = contingency_table[['Economic', 'Physical']]

print("\n--- Contingency Table (Analyzed for Hypothesis) ---")
print(contingency_table)

# 6. Statistical Test
# Given small sample sizes (likely < 5 in some cells), Fisher's Exact Test is safer than Chi-square
# Fisher's requires a 2x2 table.
if contingency_table.shape == (2, 2):
    # fisher_exact returns (odds_ratio, p_value)
    # The table is: [[Fin-Eco, Fin-Phys], [Trans-Eco, Trans-Phys]]
    # Hypothesis: Financial -> Economic, Transportation -> Physical
    # So we expect Fin-Eco to be high, Fin-Phys low.
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    test_name = "Fisher's Exact Test"
else:
    # Fallback to Chi2 if shape is weird (e.g. only one sector found)
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    test_name = "Chi-Square Test"
    odds_ratio = 0

print(f"\n{test_name} Results:")
print(f"P-value: {p_value:.4e}")
if test_name == "Fisher's Exact Test":
    print(f"Odds Ratio: {odds_ratio:.4f}")

# 7. Visualization including 'Other' to show full picture
full_contingency = pd.crosstab(subset['analyzed_sector'], subset['harm_category'])
# Normalize rows
full_pct = full_contingency.div(full_contingency.sum(axis=1), axis=0) * 100

plt.figure(figsize=(10, 6))
# Use colors: Economic=Gold, Physical=Tomato, Other=Grey
color_map = {'Economic': '#FFD700', 'Physical': '#FF6347', 'Other': '#D3D3D3'}
cols = [c for c in ['Economic', 'Physical', 'Other'] if c in full_pct.columns]
colors = [color_map.get(c, '#333333') for c in cols]

ax = full_pct[cols].plot(kind='bar', stacked=True, color=colors)
plt.title('Distribution of AI Harm Types by Sector (Description-based)')
plt.xlabel('Sector')
plt.ylabel('Percentage of Incidents')
plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center')

plt.show()

# 8. Interpretation
print("\n--- Analysis ---")
if p_value < 0.05:
    print("Result: Statistically significant relationship (p < 0.05).")
    print("The data supports the hypothesis that harm types are domain-dependent.")
    if test_name == "Fisher's Exact Test":
        if odds_ratio > 1:
            print("Positive association consistent with hypothesis (Financial biased towards Economic, Transportation towards Physical) if columns ordered [Eco, Phys].")
        else:
            print("Association observed, check directionality in table.")
else:
    print("Result: No statistically significant relationship (p >= 0.05).")
