import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import os
import sys
import traceback

# [debug]
# print("Starting experiment...")

def load_data():
    # Try current directory first, then parent
    candidates = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
    for path in candidates:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError("Could not find astalabs_discovery_all_data.csv in current or parent directory.")

try:
    # Load the dataset
    df = load_data()

    # Segment the data
    atlas_df = df[df['source_table'] == 'step3_incident_coding'].copy()
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

    print(f"ATLAS subset shape: {atlas_df.shape}")
    print(f"AIID subset shape: {aiid_df.shape}")

    # --- Analyze ATLAS (Theoretical/Research) ---
    # Identify the active tactics column
    tactic_cols = ['tactics', 'tactics_used']
    active_tactic_col = None
    for col in tactic_cols:
        if col in atlas_df.columns and atlas_df[col].notna().sum() > 0:
            active_tactic_col = col
            break
    
    atlas_total = len(atlas_df)
    atlas_evasion_hits = 0
    
    if active_tactic_col:
        # Normalize and search
        tactics_series = atlas_df[active_tactic_col].fillna('').astype(str).str.lower()
        # Search for 'evasion' or specific IDs
        evasion_mask = tactics_series.str.contains('evasion', na=False) | \
                       tactics_series.str.contains('aml.ta0007', na=False) | \
                       tactics_series.str.contains('aml.ta0005', na=False)
        atlas_evasion_hits = evasion_mask.sum()
    else:
        print("Warning: Could not identify an active tactics column in ATLAS data.")

    atlas_prop = atlas_evasion_hits / atlas_total if atlas_total > 0 else 0
    
    print(f"\nATLAS (Theoretical):")
    print(f"  Total Cases: {atlas_total}")
    print(f"  Cases with 'Evasion' tactics: {atlas_evasion_hits}")
    print(f"  Proportion: {atlas_prop:.4f}")

    # --- Analyze AIID (Real-World) ---
    # Identify the active failure column
    failure_cols = ['Known AI Technical Failure', '85'] # 85 is the index from metadata if names are stripped
    active_failure_col = None
    
    # First try exact name match
    if 'Known AI Technical Failure' in aiid_df.columns:
        active_failure_col = 'Known AI Technical Failure'
    else:
        # Try fuzzy match
        cols = [c for c in aiid_df.columns if 'Technical Failure' in str(c)]
        if cols:
            active_failure_col = cols[0]
    
    aiid_total = len(aiid_df)
    aiid_robustness_hits = 0
    
    if active_failure_col:
        # Normalize and search
        failures_series = aiid_df[active_failure_col].fillna('').astype(str).str.lower()
        # Search for 'robustness' or 'reliability'
        robustness_mask = failures_series.str.contains('robustness', na=False) | \
                          failures_series.str.contains('reliability', na=False)
        aiid_robustness_hits = robustness_mask.sum()
    else:
        print("Warning: Could not identify 'Known AI Technical Failure' column in AIID data.")

    aiid_prop = aiid_robustness_hits / aiid_total if aiid_total > 0 else 0
    
    print(f"\nAIID (Real-World):")
    print(f"  Total Incidents: {aiid_total}")
    print(f"  Incidents with 'Robustness/Reliability' failures: {aiid_robustness_hits}")
    print(f"  Proportion: {aiid_prop:.4f}")

    # --- Statistical Test ---
    stat, pval = 0.0, 1.0
    if atlas_total > 0 and aiid_total > 0:
        counts = np.array([atlas_evasion_hits, aiid_robustness_hits])
        nobs = np.array([atlas_total, aiid_total])
        
        # Two-sided Z-test
        stat, pval = proportions_ztest(counts, nobs)
        
        print(f"\n--- Statistical Comparison (Z-test) ---")
        print(f"Z-score: {stat:.4f}")
        print(f"P-value: {pval:.4e}")
        
        interpretation = "Significant difference" if pval < 0.05 else "No significant difference"
        print(f"Result: {interpretation}")
    else:
        print("\nInsufficient data for Z-test.")

    # --- Visualization ---
    labels = ['ATLAS\n(Theoretical Evasion)', 'AIID\n(Real-World Robustness)']
    proportions = [atlas_prop, aiid_prop]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, proportions, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    
    # Add exact numbers on bars
    for bar, prop in zip(bars, proportions):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                 f'{prop:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    title_text = f"The 'Theory-Reality' Gap: Evasion vs. Robustness\n(p={pval:.4e})"
    plt.title(title_text, fontsize=14)
    plt.ylabel('Prevalence (Proportion of Dataset)')
    plt.ylim(0, max(proportions) * 1.2 if max(proportions) > 0 else 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate with N
    if max(proportions) > 0:
        plt.text(0, atlas_prop/2 if atlas_prop > 0 else 0, f"n={atlas_total}", ha='center', color='white', fontweight='bold')
        plt.text(1, aiid_prop/2 if aiid_prop > 0 else 0, f"n={aiid_total}", ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
