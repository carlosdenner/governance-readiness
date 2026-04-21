import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filename, low_memory=False)
    
    # Filter for EO 13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Loaded EO 13960 dataset: {len(df_eo)} rows")

    # 1. Create 'is_biometric' (Independent Variable)
    keywords = ['face', 'facial', 'biometric', 'voice', 'recognition']
    pattern = '|'.join(keywords)
    df_eo['2_use_case_name'] = df_eo['2_use_case_name'].astype(str)
    df_eo['is_biometric'] = df_eo['2_use_case_name'].str.contains(pattern, case=False, na=False)

    # 2. Create 'has_mitigation' (Dependent Variable) using text analysis
    # Logic: Default to True, set to False if text indicates N/A, None, or No mechanism.
    def check_mitigation(text):
        if pd.isna(text):
            return False
        s = str(text).strip().lower()
        if s == '' or s == 'nan':
            return False
        
        # Negative indicators at the start or specific phrases
        if s.startswith('n/a'): return False
        if s.startswith('none'): return False
        if s.startswith('no '): return False
        if s.startswith('not '): return False
        if 'not applicable' in s: return False
        if 'no demographic' in s: return False
        if 'not safety' in s: return False
        
        # If it passed all above, we assume it describes a mitigation
        return True

    df_eo['has_mitigation'] = df_eo['62_disparity_mitigation'].apply(check_mitigation)

    # 3. Generate Contingency Table
    contingency = pd.crosstab(df_eo['is_biometric'], df_eo['has_mitigation'])
    
    # Force 2x2 shape [False, True]
    contingency = contingency.reindex(index=[False, True], columns=[False, True], fill_value=0)
    contingency.index = ['Non-Biometric', 'Biometric']
    contingency.columns = ['No Mitigation', 'Has Mitigation']

    print("\n--- Contingency Table (Counts) ---")
    print(contingency)

    # 4. Statistical Testing
    # Fisher's Exact Test
    stat, p_val = fisher_exact(contingency)
    print(f"\n--- Fisher's Exact Test Results ---")
    print(f"P-value: {p_val:.4f}")
    
    # Chi-square
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    print(f"Chi-Square Statistic: {chi2:.4f}, P-value: {p_chi2:.4f}")

    # 5. Calculate Odds Ratio & Compliance Rates
    non_bio_no = contingency.loc['Non-Biometric', 'No Mitigation']
    non_bio_yes = contingency.loc['Non-Biometric', 'Has Mitigation']
    bio_no = contingency.loc['Biometric', 'No Mitigation']
    bio_yes = contingency.loc['Biometric', 'Has Mitigation']
    
    non_bio_total = non_bio_no + non_bio_yes
    bio_total = bio_no + bio_yes
    
    print("\n--- Compliance Rates ---")
    if non_bio_total > 0:
        nb_rate = (non_bio_yes / non_bio_total) * 100
        print(f"Non-Biometric: {nb_rate:.2f}% ({non_bio_yes}/{non_bio_total})")
    
    if bio_total > 0:
        b_rate = (bio_yes / bio_total) * 100
        print(f"Biometric:     {b_rate:.2f}% ({bio_yes}/{bio_total})")

    # Odds Ratio
    # OR = (bio_yes / bio_no) / (non_bio_yes / non_bio_no)
    if bio_no > 0 and non_bio_yes > 0:
        or_val = (bio_yes * non_bio_no) / (bio_no * non_bio_yes)
        print(f"\nOdds Ratio: {or_val:.4f}")
    else:
        print("\nOdds Ratio: Undefined (division by zero)")

except Exception as e:
    print(f"An error occurred: {e}")