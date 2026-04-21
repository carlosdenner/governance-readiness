import json
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# [debug]
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

# Try loading from current directory first, as previous experiments succeeded there
file_path = 'step3_enrichments.json'
if not os.path.exists(file_path):
    # If not found, try the parent directory as per the note, though it failed last time
    file_path = '../step3_enrichments.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter for relevant harm types
    target_harms = ['supply_chain', 'reliability']
    df_filtered = df[df['harm_type'].isin(target_harms)].copy()
    
    # Calculate sub-competency count (handle potential missing or null lists)
    # The metadata indicates 'sub_competency_ids' is the field
    def get_count(x):
        if isinstance(x, list):
            return len(x)
        elif isinstance(x, str):
            # Handle case where it might be a string representation of a list or semicolon separated
            if x.startswith('['):
                try:
                    return len(eval(x))
                except:
                    return 0
            return len(x.split(';'))
        return 0

    df_filtered['gap_count'] = df_filtered['sub_competency_ids'].apply(get_count)
    
    # Separate groups
    supply_chain_group = df_filtered[df_filtered['harm_type'] == 'supply_chain']['gap_count']
    reliability_group = df_filtered[df_filtered['harm_type'] == 'reliability']['gap_count']
    
    # Descriptive Statistics
    print(f"\n=== Descriptive Statistics (N={len(df_filtered)}) ===")
    print(f"Supply Chain (n={len(supply_chain_group)}): Mean={supply_chain_group.mean():.2f}, Std={supply_chain_group.std():.2f}")
    print(f"Reliability (n={len(reliability_group)}): Mean={reliability_group.mean():.2f}, Std={reliability_group.std():.2f}")
    
    # Mann-Whitney U Test
    # Alternative 'two-sided' is standard for generic difference testing
    u_stat, p_val = stats.mannwhitneyu(supply_chain_group, reliability_group, alternative='two-sided')
    
    print("\n=== Mann-Whitney U Test Results ===")
    print(f"U-statistic: {u_stat}")
    print(f"P-value: {p_val:.4f}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([supply_chain_group, reliability_group], labels=['Supply Chain', 'Reliability'])
    plt.title('Competency Gap Breadth: Supply Chain vs Reliability')
    plt.ylabel('Number of Missing Sub-Competencies')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Jitter plot for individual points
    import numpy as np
    x_supply = np.random.normal(1, 0.04, size=len(supply_chain_group))
    x_rel = np.random.normal(2, 0.04, size=len(reliability_group))
    plt.scatter(x_supply, supply_chain_group, alpha=0.6, label='Supply Chain Cases')
    plt.scatter(x_rel, reliability_group, alpha=0.6, label='Reliability Cases')
    plt.legend()
    
    plt.show()
    
except Exception as e:
    print(f"An error occurred: {e}")
