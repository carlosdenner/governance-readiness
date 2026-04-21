import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def run_experiment():
    print("Starting Experiment: ATLAS Tactic-Gap Fingerprinting")
    
    # 1. Load Dataset
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        return

    # 2. Filter for relevant table
    target_table = 'step3_incident_coding'
    df_subset = df[df['source_table'] == target_table].copy()
    print(f"Subset shape: {df_subset.shape}")

    if len(df_subset) == 0:
        print("No data in step3_incident_coding.")
        return

    # 3. Define mapping logic
    # Tactic Codes (Based on MITRE ATLAS)
    # AML.TA0005: Defense Evasion
    # AML.TA0010: Exfiltration
    # We also check for 'Evasion' or 'Exfiltration' in text if available, but primarily codes.
    
    # Gap Keywords
    # Robustness: 'Robustness', 'Hardening', 'Adversarial Input', 'Ensemble'
    # Access Control: 'Access', 'Privilege', 'Boundary', 'Authentication'

    def parse_list(x):
        if pd.isna(x): return []
        s = str(x)
        return [i.strip() for i in s.replace(';', ',').split(',') if i.strip()]

    df_subset['tactic_codes'] = df_subset['tactics_used'].apply(parse_list)
    
    # Helper to check content
    def check_tactic(codes, target_codes):
        return any(c in codes for c in target_codes)

    def check_gap(row, keywords):
        # Check both competency_domains and missing_controls
        text_content = str(row.get('competency_domains', '')) + " " + str(row.get('missing_controls', ''))
        return any(k.lower() in text_content.lower() for k in keywords)

    # 4. Create Binary Flags
    # Evasion: AML.TA0005
    # Note: Using broad check just in case numbering differs, but TA0005 is standard ATLAS Defense Evasion
    # Also checking TA0006 just in case.
    evasion_codes = ['AML.TA0005', 'AML.TA0006'] 
    exfiltration_codes = ['AML.TA0010', 'AML.TA0011'] # TA0011 is Impact, but sometimes grouped. Sticking to TA0010 main.
    
    df_subset['is_evasion_tactic'] = df_subset['tactic_codes'].apply(lambda x: check_tactic(x, evasion_codes))
    df_subset['is_exfil_tactic'] = df_subset['tactic_codes'].apply(lambda x: check_tactic(x, exfiltration_codes))
    
    robustness_keywords = ['Robustness', 'Hardening', 'Adversarial', 'Ensemble', 'Perturbation']
    access_keywords = ['Access', 'Privilege', 'Boundary', 'Authentication', 'Credential']

    df_subset['is_robustness_gap'] = df_subset.apply(lambda r: check_gap(r, robustness_keywords), axis=1)
    df_subset['is_access_gap'] = df_subset.apply(lambda r: check_gap(r, access_keywords), axis=1)

    # 5. Statistical Testing
    print("\n--- Analysis 1: Evasion Tactics vs Robustness Gaps ---")
    ct_evasion = pd.crosstab(df_subset['is_evasion_tactic'], df_subset['is_robustness_gap'])
    print(ct_evasion)
    if ct_evasion.size == 4:
        odds_ev, p_ev = stats.fisher_exact(ct_evasion)
        print(f"Fisher p-value: {p_ev:.4f}")
        print(f"Odds Ratio: {odds_ev:.2f}")
    else:
        print("Insufficient table size for stats.")

    print("\n--- Analysis 2: Exfiltration Tactics vs Access Control Gaps ---")
    ct_exfil = pd.crosstab(df_subset['is_exfil_tactic'], df_subset['is_access_gap'])
    print(ct_exfil)
    if ct_exfil.size == 4:
        odds_ex, p_ex = stats.fisher_exact(ct_exfil)
        print(f"Fisher p-value: {p_ex:.4f}")
        print(f"Odds Ratio: {odds_ex:.2f}")
    else:
        print("Insufficient table size for stats.")

    # 6. Visualization
    # We will create a heatmap of specific Tactic Codes vs Competency Domains
    # We explode the lists to get pairs
    
    heatmap_pairs = []
    for _, row in df_subset.iterrows():
        t_list = row['tactic_codes']
        # Parse domains
        d_raw = str(row.get('competency_domains', ''))
        d_list = [d.strip() for d in d_raw.split(';') if d.strip()]
        
        for t in t_list:
            for d in d_list:
                # Shorten domain for display
                d_short = d.split('--')[-1].strip() if '--' in d else d
                heatmap_pairs.append({'Tactic Code': t, 'Competency Gap': d_short})

    if heatmap_pairs:
        df_hm = pd.DataFrame(heatmap_pairs)
        # Filter to top occurring for readability
        top_tactics = df_hm['Tactic Code'].value_counts().head(15).index
        top_gaps = df_hm['Competency Gap'].value_counts().head(15).index
        
        df_hm_filtered = df_hm[df_hm['Tactic Code'].isin(top_tactics) & df_hm['Competency Gap'].isin(top_gaps)]
        
        ct_hm = pd.crosstab(df_hm_filtered['Tactic Code'], df_hm_filtered['Competency Gap'])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(ct_hm, annot=True, fmt='d', cmap='Reds')
        plt.title('Fingerprint: Adversarial Tactics vs Competency Gaps (Top 15)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No data pairs for heatmap.")

if __name__ == "__main__":
    run_experiment()
