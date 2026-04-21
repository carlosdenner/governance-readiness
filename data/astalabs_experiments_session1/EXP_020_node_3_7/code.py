import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Load dataset
file_path = 'step3_incident_coding.csv'
# Handle potential file location differences
if not os.path.exists(file_path):
    if os.path.exists(os.path.join('..', file_path)):
        file_path = os.path.join('..', file_path)

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# --- Preprocessing ---

# 1. Parse Tactics using correct IDs
# AML.TA0010 = Exfiltration
# AML.TA0002 = Reconnaissance

df['tactics_used'] = df['tactics_used'].fillna('').astype(str)

df['has_exfiltration'] = df['tactics_used'].apply(lambda x: 1 if 'AML.TA0010' in x else 0)
df['has_reconnaissance'] = df['tactics_used'].apply(lambda x: 1 if 'AML.TA0002' in x else 0)

print("\n--- Tactic Frequency ---")
print(f"Incidents with Exfiltration (AML.TA0010): {df['has_exfiltration'].sum()}")
print(f"Incidents with Reconnaissance (AML.TA0002): {df['has_reconnaissance'].sum()}")

# 2. Parse Trust/Integration Split
# Mapping: trust-dominant -> 0, both -> 0.5, integration-dominant -> 1
# This creates a continuous 'Integration Orientation' scale.
mapping = {
    'trust-dominant': 0.0,
    'both': 0.5,
    'integration-dominant': 1.0
}

df['integration_score'] = df['trust_integration_split'].map(mapping)

# Filter out any unmapped rows (though previous check showed none)
df = df.dropna(subset=['integration_score'])

# --- Statistical Analysis ---

def analyze_tactic(name, col_name):
    # Split groups
    present = df[df[col_name] == 1]['integration_score']
    absent = df[df[col_name] == 0]['integration_score']
    
    # Check if we have enough data
    if len(present) < 2 or len(absent) < 2:
        print(f"\nSkipping statistical test for {name} due to insufficient sample size (Present: {len(present)}, Absent: {len(absent)}).")
        return None

    # T-test
    t_stat, p_val = stats.ttest_ind(present, absent, equal_var=False)
    
    print(f"\nAnalysis for {name}:")
    print(f"  Mean Integration Score (Present): {present.mean():.3f} (n={len(present)}) | Std: {present.std():.3f}")
    print(f"  Mean Integration Score (Absent):  {absent.mean():.3f} (n={len(absent)}) | Std: {absent.std():.3f}")
    print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    return {
        'tactic': name,
        'mean_present': present.mean(),
        'mean_absent': absent.mean(),
        'p_value': p_val
    }

results = []
res_exfil = analyze_tactic('Exfiltration', 'has_exfiltration')
if res_exfil: results.append(res_exfil)

res_recon = analyze_tactic('Reconnaissance', 'has_reconnaissance')
if res_recon: results.append(res_recon)

# --- Visualization ---
if results:
    plt.figure(figsize=(10, 6))

    tactics = [r['tactic'] for r in results]
    means_present = [r['mean_present'] for r in results]
    means_absent = [r['mean_absent'] for r in results]
    p_values = [r['p_value'] for r in results]

    x = np.arange(len(tactics))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means_present, width, label='Tactic Present', color='skyblue')
    rects2 = ax.bar(x + width/2, means_absent, width, label='Tactic Absent', color='lightgray')

    ax.set_ylabel('Mean Integration Score (0=Trust, 1=Integration)')
    ax.set_title('Integration Orientation by Tactic Presence')
    ax.set_xticks(x)
    ax.set_xticklabels(tactics)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # Add p-values and counts
    for i, p in enumerate(p_values):
        height = max(means_present[i], means_absent[i])
        ax.text(i, height + 0.02, f'p={p:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("No valid results to plot.")
