import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import scipy.stats as stats
import re
import os
import numpy as np

# [debug]
print("Starting experiment: ATLAS Kill Chain Analysis (Attempt 4 - Fixed Parser)")

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    raise FileNotFoundError(f"Could not find {filename}")

df = pd.read_csv(filepath, low_memory=False)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"ATLAS cases loaded: {len(atlas_df)}")

# robust parser for pipe-delimited or list-like strings
def parse_and_clean_tactics(tactic_str):
    if pd.isna(tactic_str):
        return []
    
    tactic_str = str(tactic_str).strip()
    
    # List of raw tactic tokens
    tokens = []
    
    # Check for pipe delimiter first (as seen in debug output)
    if '|' in tactic_str:
        tokens = tactic_str.split('|')
    # Check for list literal format
    elif tactic_str.startswith('[') and tactic_str.endswith(']'):
        try:
            parsed = ast.literal_eval(tactic_str)
            if isinstance(parsed, list):
                tokens = [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            # Fallback for malformed lists
            tokens = tactic_str.replace('[', '').replace(']', '').replace("'", '').replace('"', '').split(',')
    else:
        # Assume comma separated
        tokens = tactic_str.split(',')
        
    # Clean tokens
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if not token: continue
        
        # Regex to extract name from {{name.id}}
        match = re.search(r'\{\{([a-zA-Z0-9_]+)\.id\}\}', token)
        if match:
            name = match.group(1)
            # Normalize: underscores to spaces, title case
            # Handle specific mapping if needed, but Title Case is usually sufficient
            name = name.replace('_', ' ').title()
            cleaned_tokens.append(name)
        else:
            # If no curly braces, just clean the string
            # Remove potential .id suffix if it exists plainly
            name = token.replace('.id', '').replace('_', ' ').title()
            cleaned_tokens.append(name)
            
    return sorted(list(set(cleaned_tokens))) # Return unique tactics for this row

# Apply parsing
atlas_df['tactics_clean'] = atlas_df['tactics'].apply(parse_and_clean_tactics)

# Verify parsing
all_clean_tactics = set()
for tactics in atlas_df['tactics_clean']:
    all_clean_tactics.update(tactics)
all_clean_tactics = sorted(list(all_clean_tactics))

print(f"Unique tactics identified: {len(all_clean_tactics)}")
print(f"Tactics list: {all_clean_tactics}")

# Define target variables
# Hypothesis: 'Defense Evasion' (Evasion) associated with 'Exfiltration'
atlas_df['has_evasion'] = atlas_df['tactics_clean'].apply(lambda x: 'Defense Evasion' in x)
atlas_df['has_exfiltration'] = atlas_df['tactics_clean'].apply(lambda x: 'Exfiltration' in x)

# Contingency Table
contingency_table = pd.crosstab(atlas_df['has_evasion'], atlas_df['has_exfiltration'])
print("\nContingency Table (Defense Evasion vs Exfiltration):")
print(contingency_table)

# Ensure 2x2 for Fisher's test (handling missing columns/rows)
# Expected columns: False, True. Expected index: False, True.
full_contingency = pd.DataFrame(0, index=[False, True], columns=[False, True])
for i in [False, True]:
    for c in [False, True]:
        if i in contingency_table.index and c in contingency_table.columns:
            full_contingency.loc[i, c] = contingency_table.loc[i, c]

print("\nFull Contingency Table:")
print(full_contingency)

# Fisher's Exact Test
odds_ratio, p_value = stats.fisher_exact(full_contingency)
print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Significant association found (Reject H0).")
else:
    print("Conclusion: No significant association found (Fail to reject H0).")

# Co-occurrence Heatmap
if len(all_clean_tactics) > 0:
    co_occurrence = pd.DataFrame(0, index=all_clean_tactics, columns=all_clean_tactics)

    for tactics in atlas_df['tactics_clean']:
        # Permutation: count co-occurrence for every pair in the list
        for t1 in tactics:
            for t2 in tactics:
                co_occurrence.loc[t1, t2] += 1

    plt.figure(figsize=(12, 10))
    # Mask diagonal to emphasize co-occurrences? No, standard heatmap.
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Adversarial Tactic Co-occurrence Heatmap (ATLAS) - Final')
    plt.tight_layout()
    plt.show()