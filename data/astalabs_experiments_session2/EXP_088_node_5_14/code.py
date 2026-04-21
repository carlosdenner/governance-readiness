import pandas as pd
import numpy as np
import os
import math

# --- Helper Functions ---

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def z_test_proportions(count1, nobs1, count2, nobs2):
    # Calculates two-sided Z-test for proportions
    if nobs1 == 0 or nobs2 == 0:
        return 0.0, 1.0
    
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    
    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0
        
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs1 + 1/nobs2))
    
    if se == 0:
        return 0.0, 1.0
        
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm_cdf(abs(z)))
    return z, p_value

# --- Experiment Code ---

def run_experiment():
    print("Starting Agency Risk Appetite Experiment...\n")
    
    # 1. Load Dataset
    filename = 'astalabs_discovery_all_data.csv'
    filepath = filename
    if not os.path.exists(filepath):
        filepath = '../' + filename
        
    if not os.path.exists(filepath):
        print(f"Error: Dataset {filename} not found.")
        return

    print(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter for EO13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Loaded EO13960 subset: {len(df_eo)} rows")

    # 2. Categorize Agencies
    security_keywords = ['Department of Defense', 'Homeland Security', 'Department of Justice']
    social_keywords = ['Health and Human Services', 'Department of Education', 'Veterans Affairs']

    def get_category(agency):
        if pd.isna(agency):
            return None
        agency_str = str(agency)
        if any(k in agency_str for k in security_keywords):
            return 'Security'
        if any(k in agency_str for k in social_keywords):
            return 'Social'
        return 'Other'

    df_eo['Agency_Category'] = df_eo['3_agency'].apply(get_category)
    
    # Filter for analysis groups
    df_analysis = df_eo[df_eo['Agency_Category'].isin(['Security', 'Social'])].copy()
    
    print("\nAgency Category Distribution:")
    print(df_analysis['Agency_Category'].value_counts())

    if df_analysis.empty:
        print("No matching agencies found. Exiting.")
        return

    # 3. Analyze Rights-Impacting Systems
    # Robust check using vectorized string operation
    df_analysis['is_rights'] = df_analysis['17_impact_type'].astype(str).str.contains('Rights', case=False, na=False)

    print("\n--- Analysis 1: Rights-Impacting Systems ---")
    # Groupby to get counts
    rights_counts = df_analysis.groupby('Agency_Category')['is_rights'].sum()
    total_counts = df_analysis.groupby('Agency_Category')['is_rights'].count()
    
    rights_stats = pd.DataFrame({'Rights_Count': rights_counts, 'Total_Systems': total_counts})
    rights_stats['Proportion'] = rights_stats['Rights_Count'] / rights_stats['Total_Systems']
    print(rights_stats)

    # Z-test for Rights-Impacting
    if 'Security' in rights_stats.index and 'Social' in rights_stats.index:
        sec_cnt = rights_stats.loc['Security', 'Rights_Count']
        sec_tot = rights_stats.loc['Security', 'Total_Systems']
        soc_cnt = rights_stats.loc['Social', 'Rights_Count']
        soc_tot = rights_stats.loc['Social', 'Total_Systems']
        
        z1, p1 = z_test_proportions(sec_cnt, sec_tot, soc_cnt, soc_tot)
        print(f"\nZ-Test (Rights-Impacting - Security vs Social): z = {z1:.4f}, p = {p1:.4e}")
        if p1 < 0.05:
            print("Result: Significant difference.")
        else:
            print("Result: No significant difference.")
    else:
        print("Cannot perform Z-test: Missing Security or Social group.")

    # 4. Analyze Independent Evaluation (Subset: Rights-Impacting)
    df_rights_subset = df_analysis[df_analysis['is_rights']].copy()
    
    # Robust check for 'Yes' in '55_independent_eval'
    df_rights_subset['has_eval'] = df_rights_subset['55_independent_eval'].astype(str).str.contains('Yes', case=False, na=False)

    print("\n--- Analysis 2: Independent Evaluation (Rights-Impacting Subset) ---")
    if len(df_rights_subset) == 0:
        print("No Rights-Impacting systems found to analyze.")
    else:
        eval_counts = df_rights_subset.groupby('Agency_Category')['has_eval'].sum()
        eval_total = df_rights_subset.groupby('Agency_Category')['has_eval'].count()
        
        eval_stats = pd.DataFrame({'Eval_Yes_Count': eval_counts, 'Total_Rights_Systems': eval_total})
        eval_stats['Proportion'] = eval_stats['Eval_Yes_Count'] / eval_stats['Total_Rights_Systems']
        print(eval_stats)

        # Z-test for Independent Eval
        if 'Security' in eval_stats.index and 'Social' in eval_stats.index:
            sec_e_cnt = eval_stats.loc['Security', 'Eval_Yes_Count']
            sec_e_tot = eval_stats.loc['Security', 'Total_Rights_Systems']
            soc_e_cnt = eval_stats.loc['Social', 'Eval_Yes_Count']
            soc_e_tot = eval_stats.loc['Social', 'Total_Rights_Systems']
            
            if sec_e_tot > 0 and soc_e_tot > 0:
                z2, p2 = z_test_proportions(sec_e_cnt, sec_e_tot, soc_e_cnt, soc_e_tot)
                print(f"\nZ-Test (Independent Eval - Security vs Social): z = {z2:.4f}, p = {p2:.4e}")
                if p2 < 0.05:
                    print("Result: Significant difference.")
                else:
                    print("Result: No significant difference.")
            else:
                 print("Cannot perform Z-test: One group has 0 systems.")
        else:
            print("Insufficient categories for comparison in subset.")

if __name__ == "__main__":
    run_experiment()