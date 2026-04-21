import pandas as pd
import scipy.stats as stats
import numpy as np

try:
    # Load dataset
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    
    # Filter for AIID incidents
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    
    # --- Preprocessing Autonomy Level ---
    def map_autonomy(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val == 'Autonomy3':
            return 'High'
        elif val in ['Autonomy1', 'Autonomy2']:
            return 'Low'
        return np.nan

    aiid['autonomy_bin'] = aiid['Autonomy Level'].apply(map_autonomy)
    
    # --- Preprocessing Harm Severity ---
    def map_severity(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val == 'AI tangible harm event':
            return 'Severe'
        elif val in ['AI tangible harm near-miss', 'AI tangible harm issue', 'none']:
            return 'Not Severe'
        return np.nan

    aiid['severity_bin'] = aiid['AI Harm Level'].apply(map_severity)
    
    # Drop rows where either variable is NaN
    analysis_df = aiid.dropna(subset=['autonomy_bin', 'severity_bin'])
    
    print(f"Data points for analysis: {len(analysis_df)}")
    
    # Contingency Table
    contingency = pd.crosstab(analysis_df['autonomy_bin'], analysis_df['severity_bin'])
    print("\n--- Contingency Table (Autonomy vs Severity) ---")
    print(contingency)
    
    if contingency.shape == (2, 2):
        # Chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
        print(f"\nChi-square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        
        # Odds Ratio Calculation
        # Standard OR formula: (a*d) / (b*c)
        # Where a = High+Severe, b = High+NotSevere, c = Low+Severe, d = Low+NotSevere
        # However, crosstab order depends on sorting. Let's extract explicitly.
        
        try:
            n_high_severe = contingency.loc['High', 'Severe']
            n_high_not = contingency.loc['High', 'Not Severe']
            n_low_severe = contingency.loc['Low', 'Severe']
            n_low_not = contingency.loc['Low', 'Not Severe']
            
            if n_high_not * n_low_severe == 0:
                odds_ratio = np.inf
            else:
                odds_ratio = (n_high_severe * n_low_not) / (n_high_not * n_low_severe)
                
            print(f"Odds Ratio (High Autonomy -> Severe Harm): {odds_ratio:.4f}")
            
        except KeyError as e:
            print(f"Could not calculate OR due to missing keys in contingency table: {e}")
    else:
        print("\nContingency table is not 2x2. Cannot perform standard binary Odds Ratio calculation.")

except Exception as e:
    print(f"An error occurred: {e}")
