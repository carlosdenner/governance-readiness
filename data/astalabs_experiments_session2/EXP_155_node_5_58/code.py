import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import re

def load_data():
    filename = 'astalabs_discovery_all_data.csv'
    # Try current directory first, then parent directory
    if os.path.exists(filename):
        return pd.read_csv(filename, low_memory=False)
    elif os.path.exists(os.path.join('..', filename)):
        return pd.read_csv(os.path.join('..', filename), low_memory=False)
    else:
        raise FileNotFoundError(f"{filename} not found in current or parent directory.")

def run_experiment():
    try:
        df = load_data()
        print("Dataset loaded successfully.")
    except Exception as e:
        print(e)
        return

    # Filter for ATLAS cases
    atlas = df[df['source_table'] == 'atlas_cases'].copy()
    print(f"ATLAS cases found: {len(atlas)}")

    if len(atlas) == 0:
        print("No ATLAS cases found. Check dataset.")
        return

    # Text Analysis to classify Target Sector
    # Keywords for Government/Defense
    gov_keywords = [
        'government', 'defense', 'military', 'federal', 'agency', 'state', 
        'national', 'intelligence', 'surveillance', 'police', 'election', 
        'voting', 'public sector', 'ministry', 'army', 'navy', 'air force'
    ]

    def classify_sector(row):
        text = (str(row.get('name', '')) + " " + str(row.get('summary', ''))).lower()
        if any(kw in text for kw in gov_keywords):
            return 'Government/Defense'
        return 'Commercial/Private'

    atlas['sector'] = atlas.apply(classify_sector, axis=1)

    # Calculate Tactic Chain Length
    # The 'tactics' column might be comma separated or list-like. 
    # We'll count distinct items.
    def count_tactics(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        # Remove brackets if they exist (JSON style) and split
        s = str(val).replace('[', '').replace(']', '').replace("'", "")
        # Split by comma
        items = [i.strip() for i in s.split(',') if i.strip()]
        return len(set(items)) # distinct tactics

    atlas['tactic_count'] = atlas['tactics'].apply(count_tactics)

    # Group Data
    gov_counts = atlas[atlas['sector'] == 'Government/Defense']['tactic_count']
    com_counts = atlas[atlas['sector'] == 'Commercial/Private']['tactic_count']

    print(f"\nSector Analysis:\n  Government/Defense: n={len(gov_counts)}, Mean Tactic Count={gov_counts.mean():.2f}\n  Commercial/Private: n={len(com_counts)}, Mean Tactic Count={com_counts.mean():.2f}")

    # Statistical Test
    # Using Mann-Whitney U test (non-parametric) as counts are often not normal
    # and sample sizes might be small.
    stat, p_val = stats.mannwhitneyu(gov_counts, com_counts, alternative='greater')
    
    print(f"\nMann-Whitney U Test (Gov > Com):\n  U-statistic={stat}\n  p-value={p_val:.4f}")

    # Independent T-test for robustness check
    t_stat, t_p = stats.ttest_ind(gov_counts, com_counts, equal_var=False)
    print(f"T-test (two-sided): t={t_stat:.4f}, p={t_p:.4f}")

    # Interpretation
    alpha = 0.05
    if p_val < alpha:
        print("\nResult: Government/Defense targets involve significantly longer tactic chains.")
    else:
        print("\nResult: No significant difference in tactic chain length between sectors.")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([gov_counts, com_counts], labels=['Gov/Def', 'Com/Priv'], patch_artist=True)
    plt.title('Attack Sophistication: Tactic Counts by Target Sector')
    plt.ylabel('Number of Distinct Tactics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    run_experiment()