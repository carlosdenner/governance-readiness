import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    # 1. Load Dataset
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        filepath = filename
    elif os.path.exists(os.path.join('..', filename)):
        filepath = os.path.join('..', filename)
    else:
        print(f"Dataset {filename} not found.")
        return

    print(f"Loading {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Failed to load csv: {e}")
        return

    # 2. Filter for ATLAS cases
    # The metadata indicates 'atlas_cases' or 'step3_incident_coding'.
    # We prefer 'step3_incident_coding' as it likely contains the coded analysis, 
    # but 'atlas_cases' is the raw source. We'll check both.
    
    # Let's try to get a consolidated view or pick the best source.
    # Looking at previous exploration, 'step3_incident_coding' has 52 rows and 'tactics' column.
    
    atlas_df = df[df['source_table'].isin(['atlas_cases', 'step3_incident_coding'])].copy()
    
    if atlas_df.empty:
        print("No ATLAS case rows found.")
        return

    # 3. Identify the correct tactics column
    # We explicitly exclude 'n_tactics' (count) and look for string columns.
    potential_cols = ['tactics', 'tactics_used', '92_tactics'] # 92_tactics from metadata description
    
    target_col = None
    for col in potential_cols:
        if col in atlas_df.columns:
            # Check if it has non-null values and is object/string type
            valid_rows = atlas_df[col].dropna()
            if not valid_rows.empty:
                # Check if values look like strings (not numbers)
                sample = str(valid_rows.iloc[0])
                if not sample.isdigit():
                    target_col = col
                    break
    
    # Fallback: search all columns with 'tactic' in name if specific ones fail
    if not target_col:
        cols = [c for c in atlas_df.columns if 'tactic' in c.lower() and 'n_' not in c.lower()]
        for col in cols:
             valid_rows = atlas_df[col].dropna()
             if not valid_rows.empty and not str(valid_rows.iloc[0]).isdigit():
                 target_col = col
                 break

    if not target_col:
        print("Could not find a valid string column for tactics.")
        print("Available columns:", atlas_df.columns.tolist())
        return

    print(f"Using column '{target_col}' for analysis (n={atlas_df[target_col].notna().sum()}).")
    
    # 4. Parse Tactics
    # Normalize to lowercase and split
    def parse_tactics(val):
        if pd.isna(val):
            return []
        val = str(val).lower()
        # Replace common delimiters
        val = val.replace(';', ',').replace('/', ',')
        tokens = [t.strip() for t in val.split(',')]
        return tokens

    atlas_df['parsed_tactics'] = atlas_df[target_col].apply(parse_tactics)

    # 5. Create Indicators
    # We look for keywords: 'exfiltration', 'evasion' (often 'defense evasion'), 'impact'
    # Note: MITRE ATLAS uses 'Defense Evasion', so we search for 'evasion'.
    # 'Impact' is a top-level tactic.
    # 'Exfiltration' is a top-level tactic.
    
    atlas_df['has_exfil'] = atlas_df['parsed_tactics'].apply(lambda x: any('exfiltration' in t for t in x))
    atlas_df['has_evasion'] = atlas_df['parsed_tactics'].apply(lambda x: any('evasion' in t for t in x))
    atlas_df['has_impact'] = atlas_df['parsed_tactics'].apply(lambda x: any('impact' in t for t in x))

    # 6. Analysis Groups
    group_exfil = atlas_df[atlas_df['has_exfil']]
    group_impact = atlas_df[atlas_df['has_impact']]
    
    n_exfil = len(group_exfil)
    n_impact = len(group_impact)
    
    print(f"Total analyzed cases: {len(atlas_df)}")
    print(f"Cases with Exfiltration: {n_exfil}")
    print(f"Cases with Impact: {n_impact}")

    if n_exfil == 0 or n_impact == 0:
        print("Insufficient data in one or both groups to compare.")
        return

    # Count Evasion in each group
    k_exfil_evasion = group_exfil['has_evasion'].sum()
    k_impact_evasion = group_impact['has_evasion'].sum()
    
    rate_exfil = k_exfil_evasion / n_exfil
    rate_impact = k_impact_evasion / n_impact
    
    print(f"\n--- Co-occurrence Results ---")
    print(f"Exfiltration Cases: {k_exfil_evasion}/{n_exfil} ({rate_exfil:.2%}) also use Evasion")
    print(f"Impact Cases:       {k_impact_evasion}/{n_impact} ({rate_impact:.2%}) also use Evasion")

    # Fisher's Exact Test
    # Table:
    #             | Evasion+ | Evasion-
    # Exfil Group |    a     |    b
    # Impact Group|    c     |    d
    
    a = k_exfil_evasion
    b = n_exfil - k_exfil_evasion
    c = k_impact_evasion
    d = n_impact - k_impact_evasion
    
    odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
    
    print(f"\nFisher's Exact Test (H1: Exfil > Impact for Evasion co-occurrence)")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Jaccard Similarity (Exfiltration, Evasion)
    # Intersection = Cases with BOTH Exfil AND Evasion
    # Union = Cases with EITHER Exfil OR Evasion
    n_inter = atlas_df[atlas_df['has_exfil'] & atlas_df['has_evasion']].shape[0]
    n_union = atlas_df[atlas_df['has_exfil'] | atlas_df['has_evasion']].shape[0]
    jaccard = n_inter / n_union if n_union > 0 else 0
    
    print(f"\nJaccard Similarity (Exfiltration <-> Evasion): {jaccard:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Exfiltration Cases', 'Impact Cases'], [rate_exfil, rate_impact], 
                   color=['#4c72b0', '#c44e52'])
    plt.ylabel('Proportion Involving Evasion Tactics')
    plt.title('Adversarial Tech Sophistication: Stealth vs. Destruction')
    plt.ylim(0, 1.1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom')
    
    plt.show()

if __name__ == '__main__':
    run_experiment()