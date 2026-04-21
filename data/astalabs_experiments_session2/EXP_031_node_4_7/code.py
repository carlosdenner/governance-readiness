import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re

def run_experiment():
    # Load dataset
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

    # Filter for EO13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 records loaded: {len(df_eo)}")

    # Helper to find column names
    def find_col(keyword, columns):
        matches = [c for c in columns if keyword.lower() in c.lower()]
        return matches[0] if matches else None

    col_stage = find_col('dev_stage', df_eo.columns)
    col_disparity = find_col('disparity_mitigation', df_eo.columns)
    col_stakeholder = find_col('stakeholder_consult', df_eo.columns)

    if not (col_stage and col_disparity and col_stakeholder):
        print("Error: Could not identify one or more necessary columns.")
        return

    # --- 1. Define Cohorts ---
    def map_cohort(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower()
        if any(x in s for x in ['use', 'operation', 'production', 'maintenance', 'sustainment', 'implemented', 'retired']):
            return 'Legacy'
        if any(x in s for x in ['dev', 'plan', 'pilot', 'research', 'test', 'acquisition', 'initiated']):
            return 'New'
        return 'Other'

    df_eo['cohort'] = df_eo[col_stage].apply(map_cohort)
    
    # --- 2. Intelligent Scoring Logic ---
    
    def score_disparity(text):
        if pd.isna(text):
            return 0
        s = str(text).lower().strip()
        
        # Explicit negatives
        if s in ['nan', 'none', 'n/a']:
            return 0
        if s.startswith('none ') or s.startswith('n/a') or s.startswith('no ') or s.startswith('not '):
            # Check if it's a "soft" negative (e.g. "None, but...") vs hard negative
            # For now, treat starting with these as 0 to be safe, unless it contains strong positive overrides later?
            # Actually, "None for liveness... For facial verification, ICE leverages..." -> This is mixed.
            # Let's use a keyword search for positive ACTION.
            pass

        # Positive Action Keywords
        keywords = ['test', 'eval', 'monitor', 'audit', 'assess', 'review', 'check', 
                    'mitigat', 'analy', 'ensure', 'prevent', 'balanc', 'tuning', 'detect']
        
        has_action = any(k in s for k in keywords)
        
        # Negative phrases that might contain positive words (e.g. "No analysis", "Not tested")
        is_negative_statement = (
            s.startswith('no ')
            or s.startswith('not ')
            or s.startswith('n/a')
            or s.startswith('none')
            or "waived" in s
            or "not applicable" in s
        )
        
        # Heuristic: If it has action words AND isn't a primary negative statement, score 1.
        # If it starts with negative but contains "however" or "inherits" or "leverages", maybe 1.
        # Let's keep it simple: Action word present = 1, unless dominated by negative start.
        
        if has_action and not is_negative_statement:
            return 1
        # specific overrides for complex sentences seen in data
        if "inherits" in s or "leverages" in s or "working with" in s:
            return 1
            
        return 0

    def score_consultation(text):
        if pd.isna(text):
            return 0
        s = str(text).lower().strip()
        
        # 63_stakeholder_consult often contains specific checkbox labels
        # Strong negatives
        if "none of the above" in s or "waived" in s or s.startswith("n/a") or s == "none":
            return 0
            
        # Positive indicators
        keywords = ['user', 'public', 'feedback', 'comment', 'hearing', 'meeting', 
                    'union', 'labor', 'consult', 'survey', 'interview', 'test']
        
        if any(k in s for k in keywords):
            return 1
        return 0

    df_eo['score_disparity'] = df_eo[col_disparity].apply(score_disparity)
    df_eo['score_stakeholder'] = df_eo[col_stakeholder].apply(score_consultation)
    
    # Aggregate Score
    df_eo['Equity_Compliance_Score'] = df_eo['score_disparity'] + df_eo['score_stakeholder']

    # Filter Analysis Set
    df_analysis = df_eo[df_eo['cohort'].isin(['Legacy', 'New'])].copy()
    
    legacy_grp = df_analysis[df_analysis['cohort'] == 'Legacy']
    new_grp = df_analysis[df_analysis['cohort'] == 'New']
    
    # --- 3. Statistics ---
    print("\n--- Analysis Results ---")
    print(f"Legacy Cohort: n={len(legacy_grp)}")
    print(f"New Cohort:    n={len(new_grp)}")
    
    l_mean = legacy_grp['Equity_Compliance_Score'].mean()
    n_mean = new_grp['Equity_Compliance_Score'].mean()
    
    print(f"Legacy Mean Score: {l_mean:.4f}")
    print(f"New Mean Score:    {n_mean:.4f}")
    
    # T-test
    t_stat, p_val = stats.ttest_ind(legacy_grp['Equity_Compliance_Score'], 
                                    new_grp['Equity_Compliance_Score'], 
                                    equal_var=False)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4e}")
    
    if p_val < 0.05:
        print("Result: Statistically Significant Difference")
    else:
        print("Result: No Statistically Significant Difference")

    # --- 4. Detailed Control Breakdown ---
    controls = ['score_disparity', 'score_stakeholder']
    labels = ['Disparity Mitigation', 'Stakeholder Consult']
    
    l_rates = [legacy_grp[c].mean() for c in controls]
    n_rates = [new_grp[c].mean() for c in controls]
    
    print("\nControl Adoption Rates:")
    for lbl, lr, nr in zip(labels, l_rates, n_rates):
        print(f"{lbl}: Legacy={lr:.1%}, New={nr:.1%}")

    # --- 5. Visualization ---
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width/2, l_rates, width, label='Legacy (Operational)', color='#4e79a7')
    r2 = ax.bar(x + width/2, n_rates, width, label='New (Dev/Planned)', color='#f28e2b')
    
    ax.set_ylabel('Adoption Rate')
    ax.set_title('The Legacy Governance Gap: Equity Control Adoption')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    ax.bar_label(r1, fmt='%.2f', padding=3)
    ax.bar_label(r2, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()