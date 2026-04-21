import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re

def run_experiment():
    # Load dataset
    file_path = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Dataset not found at {file_path}")
        return

    # Filter for ATLAS cases
    df_atlas = df[df['source_table'] == 'atlas_cases'].copy()
    
    # Identify correct columns for tactics and techniques
    # Based on metadata, likely 'tactics' and 'techniques'
    tactic_col = None
    tech_col = None
    
    possible_tactic_cols = ['tactics', '92_tactics', 'tactics_used', '92_tactics_used']
    possible_tech_cols = ['techniques', '93_techniques', 'techniques_used', '93_techniques_used']
    
    for col in possible_tactic_cols:
        if col in df.columns:
            tactic_col = col
            break
            
    for col in possible_tech_cols:
        if col in df.columns:
            tech_col = col
            break
    
    # Fallback to incident coding if atlas_cases is empty or columns missing
    if df_atlas.empty or not tactic_col or df_atlas[tactic_col].isnull().all():
        print("Primary subset 'atlas_cases' missing or empty. Checking 'step3_incident_coding'...")
        df_atlas = df[df['source_table'] == 'step3_incident_coding'].copy()
        # re-check columns if necessary, though they should be the same in a concatenated CSV

    if not tactic_col or not tech_col:
        print("Could not identify tactic/technique columns.")
        print("Available columns:", df.columns.tolist())
        return
        
    print(f"Using columns: '{tactic_col}' and '{tech_col}'")

    # Drop rows with missing values in key columns
    df_atlas = df_atlas.dropna(subset=[tactic_col, tech_col])
    
    if df_atlas.empty:
        print("No valid data rows found with populated tactics and techniques.")
        return

    # Function to parse distinct technique counts
    def get_technique_count(text):
        if not isinstance(text, str): return 0
        # Split by comma or semicolon
        techniques = re.split(r'[,;]', text)
        # Filter empty strings and strip whitespace
        techniques = [t.strip() for t in techniques if t.strip()]
        return len(set(techniques))

    # Function to check tactic presence
    def has_tactic(text, tactic):
        if not isinstance(text, str): return False
        return tactic.lower() in text.lower()

    # Apply processing
    df_atlas['tech_count'] = df_atlas[tech_col].apply(get_technique_count)
    df_atlas['is_impact'] = df_atlas[tactic_col].apply(lambda x: has_tactic(x, 'Impact'))
    df_atlas['is_exfil'] = df_atlas[tactic_col].apply(lambda x: has_tactic(x, 'Exfiltration'))

    # Define Groups
    # Group A: Resulting in Impact (Any case with Impact)
    group_impact = df_atlas[df_atlas['is_impact']]['tech_count']
    
    # Group B: Resulting ONLY in Exfiltration (Exfiltration present, Impact absent)
    group_exfil_only = df_atlas[df_atlas['is_exfil'] & (~df_atlas['is_impact'])]['tech_count']

    # Summary Stats
    print(f"\nAnalysis of Attack Complexity (Distinct Techniques):")
    print(f"Impact Cases (n={len(group_impact)}): Mean = {group_impact.mean():.2f}, Median = {group_impact.median()}")
    print(f"Exfil-Only Cases (n={len(group_exfil_only)}): Mean = {group_exfil_only.mean():.2f}, Median = {group_exfil_only.median()}")

    if len(group_impact) < 2 or len(group_exfil_only) < 2:
        print("Insufficient sample size for statistical testing.")
        return

    # Statistical Test 
    # Using Mann-Whitney U test (non-parametric)
    # Alternative 'greater': Impact > Exfil Only
    stat, p_val = stats.mannwhitneyu(group_impact, group_exfil_only, alternative='greater')
    
    print(f"\nMann-Whitney U Test Results (Alternative: Impact > Exfil Only):")
    print(f"U-statistic: {stat}")
    print(f"P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Conclusion: Reject Null Hypothesis. Impact attacks involve significantly more techniques than Exfiltration-only attacks.")
    else:
        print("Conclusion: Fail to Reject Null Hypothesis.")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    all_counts = pd.concat([group_impact, group_exfil_only])
    if not all_counts.empty:
        max_val = all_counts.max()
        bins = range(0, int(max_val) + 3)
        
        plt.hist(group_impact, bins=bins, alpha=0.5, label='Impact Cases', density=True, color='red')
        plt.hist(group_exfil_only, bins=bins, alpha=0.5, label='Exfil-Only Cases', density=True, color='blue')
        
        plt.axvline(group_impact.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean Impact ({group_impact.mean():.1f})')
        plt.axvline(group_exfil_only.mean(), color='blue', linestyle='dashed', linewidth=1, label=f'Mean Exfil ({group_exfil_only.mean():.1f})')
        
        plt.xlabel('Count of Distinct Techniques')
        plt.ylabel('Density')
        plt.title('Kill Chain Complexity: Impact vs Exfiltration-Only')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.show()

if __name__ == "__main__":
    run_experiment()